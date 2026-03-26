import logging
import pycat
from dataclasses import dataclass, field
from typing import List
import time
from functools import wraps

@dataclass
class TimeInfo:
    arrival_time: float = None
    dequeue_time: float = None
    generated_token_time: List = field(default_factory=list)
    first_token_time: float = None
    finished_time: float = None

class CatReporter:
    def __init__(self, model_name, model_type):
        appkey = self._get_host_appkey()
        print(f"Cat: init cat for appkey: {appkey}", flush=True)
        pycat.Cat.init_cat(appkey)
        self.model_name = model_name
        self.model_type = model_type
    
    def _get_host_appkey(self):
        default_appkey = "mllm.infer.trion.test"
        try:
            file_path = '/data/webapps/appkeys'
            with open(file_path, 'r') as file:
                appkey = file.readline().strip()
                if not appkey:
                    appkey = default_appkey
        except FileNotFoundError:
            logging.warning(f'File {file_path} does not exist.')
            appkey = default_appkey
        return appkey

    def report_with_duration(self, trans_type, trans_name, duration):
        trans = pycat.Cat.new_transaction(trans_type, trans_name)
        trans.duration = duration
        trans.complete()
    
    def report_duration_to_raptor(self, time_info: TimeInfo):
        pass
    
    def report_value_to_raptor(self, value_name, value):
        trans_type = f'{self.model_type}.{self.model_name}'
        self.report_with_duration(trans_type, value_name, value * 1000)
    
    def report_error_to_raptor(self, error_code, error_message):
        error_type = f'{self.model_type}.{self.model_name}.{error_code}'
        pycat.Cat.log_error(error_type, error_message)
    
    def report_event_to_raptor(self, event_name):
        event_type = f'{self.model_type}.{self.model_name}'
        pycat.Cat.log_event(event_type, event_name)

class CatReporterOmniModel(CatReporter):
    def report_duration_to_raptor(self, time_info: TimeInfo):
        trans_type = f'{self.model_type}.{self.model_name}'
        if time_info.arrival_time is None:
            return
        output_len = len(time_info.generated_token_time)
        self.report_with_duration(trans_type, "OutputLen", output_len * 1000)
        if time_info.dequeue_time is not None:
            queue_time = int((time_info.dequeue_time - time_info.arrival_time) * 1000 * 1000)
            self.report_with_duration(trans_type, "Queue.Time", queue_time)
        if output_len > 0:
            first1token_time = int((time_info.generated_token_time[0] - time_info.arrival_time) * 1000 * 1000)
            self.report_with_duration(trans_type, "First1Token.Time", first1token_time)
            perprocess_prefill_time = int((time_info.generated_token_time[0] - time_info.dequeue_time) * 1000 * 1000)
            self.report_with_duration(trans_type, "PreprocessAndPrefill.Time", perprocess_prefill_time)
            request_time = int((time_info.generated_token_time[-1] - time_info.arrival_time) * 1000 * 1000)
            self.report_with_duration(trans_type, "Request.Time", request_time)
        if output_len > 1:
            tpot = int((time_info.generated_token_time[-1] - time_info.generated_token_time[0]) * 1000 * 1000 / (output_len - 1))
            self.report_with_duration(trans_type, "TPOT.Time", tpot)

class CatReporterVLModel(CatReporter):
    def _check_time_info(self, time_info: TimeInfo):
        if time_info.arrival_time is None:
            time_info.finished_time = time_info.first_token_time = time_info.arrival_time = 0
            return
        if time_info.finished_time is None:
            time_info.finished_time = time_info.arrival_time
        if self.model_type == "mm2t" and time_info.first_token_time is None:
            time_info.first_token_time = time_info.arrival_time
    
    def report_duration_to_raptor(self, time_info: TimeInfo):
        self._check_time_info(time_info)
        trans_type = f'{self.model_type}.{self.model_name}'
        request_time = int((time_info.finished_time - time_info.arrival_time) * 1000 * 1000)
        self.report_with_duration(trans_type, "RequestTime", request_time)
        if self.model_type == "mm2t":
            first1token_time = int((time_info.first_token_time - time_info.arrival_time) * 1000 * 1000)
            self.report_with_duration(trans_type, "First1Token.Time", first1token_time)

class EmptyCatReporter(CatReporter): 
    
    def __init__(self, model_name, model_type, model_version):
        pass
    
    def report_with_duration(self, trans_type, trans_name, duration):
        pass
    
    def report_duration_to_raptor(self, time_info: TimeInfo):
        pass
    
    def report_value_to_raptor(self, value_name, value):
        pass
    
    def report_error_to_raptor(self, error_code, error_message):
        pass
    
    def report_event_to_raptor(self, event_name):
        pass

def raptor_report(metric_name):
    """装饰器：统计方法耗时并上报cat_reporter"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            end_time = time.time()
            if hasattr(self, 'cat_reporter') and self.cat_reporter is not None:
                elapsed_ms = int((end_time - start_time) * 1000)
                self.cat_reporter.report_value_to_raptor(metric_name, elapsed_ms)
            return result
        return wrapper
    return decorator