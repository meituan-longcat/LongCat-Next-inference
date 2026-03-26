from typing import Any, Optional, Dict, Type, TypeVar
import json

T = TypeVar("T", bound="DictData")


class DictData:
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        # 手动获取类的字段名
        valid_keys = cls.__annotations__.keys()

        # 过滤数据
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}

        # 创建实例
        instance = cls()
        for k, v in filtered_data.items():
            setattr(instance, k, v)
        return instance

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

    def remove_none(self) -> None:
        instance_dict = self.__dict__
        print(instance_dict)
        keys_to_delete = [k for k, v in instance_dict.items() if v is None]
        for k in keys_to_delete:
            # https://stackoverflow.com/questions/25054729/delattr-on-class-instance-produces-unexpected-attributeerror
            delattr(type(self), k)

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


if __name__ == "__main__":

    class RawInput(DictData):
        image: str = None
        question: str

    raw_input_dict = {"question": "AAA"}
    raw_input = RawInput.from_dict(raw_input_dict)
    raw_input.remove_none()
    print(raw_input)
    print(raw_input.to_dict())
