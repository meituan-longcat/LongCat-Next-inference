from enum import Enum, auto
import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .special_token import get_spt
from utils.logger import logger


@dataclass
class StateMachineInput:
    text_id: int = None
    multi_ids: List[int] = None
    gen_image: bool = False
    gen_audio: bool = False


class StateEnum(Enum):
    INIT = auto()
    ABORT = auto()
    GEN_TEXT_STAGE = auto()
    GEN_IMAGE_STAGE = auto()
    GEN_AUDIO_STAGE = auto()
    NEXT_AUDIO_STAGE = auto()


class SmContext:
    def __init__(self, max_gen=50):
        self.gen_step = 0
        self.max_gen = max_gen
        self.audio_start = False
        self.text_end = False


class StateBase:
    def on_enter(self, context: SmContext) -> None:
        pass

    def on_exit(self, context: SmContext) -> None:
        pass

    def handle(self, input: StateMachineInput, context: SmContext) -> Optional[StateEnum]:
        raise NotImplementedError("")


# State registry to maintain the mapping
_STATE_REGISTRY: Dict[StateEnum, type] = {}


def bind(state_type: StateEnum):
    def decorator(cls):
        _STATE_REGISTRY[state_type] = cls
        return cls

    return decorator


@bind(StateEnum.INIT)
class InitState(StateBase):
    def handle(self, input: StateMachineInput, context: SmContext) -> Optional[StateEnum]:
        if input.gen_image:
            return StateEnum.GEN_IMAGE_STAGE
        elif input.gen_audio:
            return StateEnum.GEN_AUDIO_STAGE
        else:
            return StateEnum.GEN_TEXT_STAGE

@bind(StateEnum.GEN_TEXT_STAGE)
class GenTextFreeState(StateBase):
    def handle(self, input: StateMachineInput, context: SmContext) -> Optional[StateEnum]:
        return None

# ============Image============
@bind(StateEnum.GEN_IMAGE_STAGE)
class GenImageStageStage(StateBase):
    def on_enter(self, context: SmContext):
        context.gen_step = 0
    
    def handle(self, input: StateMachineInput, context: SmContext) -> Optional[StateEnum]:
        context.gen_step += 1
        # if context.gen_step == 1024:
        #     return StateEnum.GEN_IMAGE_STAGE_1
        # return None
        if input.multi_ids[0] == get_spt().IMAGE_END_TOKEN_ID:
            # return StateEnum.IMG_END
            return StateEnum.ABORT
        return None
    
# ============AUDIO============
@bind(StateEnum.GEN_AUDIO_STAGE)
class GenAudioStageStage(StateBase):
    def on_enter(self, context: SmContext):
        context.gen_step = 0
        context.audio_start = False # 此时不生成音频
        context.text_end = False # 此时不结束文本生成
    
    def handle(self, input: StateMachineInput, context: SmContext) -> Optional[StateEnum]:
        context.gen_step += 1
        # print("context.gen_step",context.gen_step,"input.multi_ids",input.multi_ids)
        if context.gen_step == context.max_gen:
            print("max_gen")
            return StateEnum.NEXT_AUDIO_STAGE
        if context.audio_start and input.multi_ids[0] == get_spt().AUDIO_END_FLAG_ID: # 8192
            print("end input.multi_ids", input.multi_ids)
            return StateEnum.NEXT_AUDIO_STAGE
        return None
    
@bind(StateEnum.NEXT_AUDIO_STAGE)
class NextAudioStageStage(StateBase):
    def handle(self, input: StateMachineInput, context: SmContext) -> Optional[StateEnum]:
        if input.text_id == get_spt().EOS_ID:
            print("gen end!")
            return StateEnum.ABORT
        elif input.text_id == get_spt().AUDIO_GEN_START_TOKEN_ID: # 继续下一阶段的生成
            print("next round gen start!")
            return StateEnum.GEN_AUDIO_STAGE
        else: # 理论上不应该生成eos和ags之外的token，这里兜底
            print(f"生成了预期之外的token{input.text_id=}")
            return StateEnum.GEN_AUDIO_STAGE

@bind(StateEnum.ABORT)
class AbortState(StateBase):
    def handle(self, input: StateMachineInput, context: SmContext) -> Optional[StateEnum]:
        return None


# ============Main State Machine============
class StateMachine:
    def __init__(self, max_gen=50):
        self.cur_state_enum: StateEnum = StateEnum.INIT
        self.context = SmContext(max_gen)
        self._states = {state_type: cls() for state_type, cls in _STATE_REGISTRY.items()}

    def transition(self, new_state_enum: StateEnum) -> bool:
        self._states[self.cur_state_enum].on_exit(self.context)
        old_state = self.cur_state_enum
        self.cur_state_enum = new_state_enum
        self._states[self.cur_state_enum].on_enter(self.context)
        logger.trace(f"\033[34m[Transition: {old_state.name} -> {self.cur_state_enum.name}]\033[0m")

    def process(self, input: StateMachineInput) -> bool:
        next_state = self._states[self.cur_state_enum].handle(input, self.context)
        if next_state is not None:
            return self.transition(next_state)
        return False

    def get_state(self) -> StateEnum:
        return self.cur_state_enum

    def get_step(self):
        return self.context.step

    def to_string(self):
        return f"{self.cur_state_enum}, {self.context.__dict__}"
