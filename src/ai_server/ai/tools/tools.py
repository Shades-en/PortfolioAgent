from abc import ABC, ABCMeta
from typing import List
from ai_server.schemas.tools import ToolArguments
from pydantic import BaseModel, Field

class RequireArgClassMeta(ABCMeta):
    def __new__(mcs, name, bases, namespace):
        if name != "Tool":  # Don't enforce on the abstract class itself
            if "Arguments" not in namespace or not isinstance(namespace["Arguments"], type):
                raise TypeError(f"{name} must define an inner pydantic class named 'Arguments'")
        return super().__new__(mcs, name, bases, namespace)

class Tool(ABC, metaclass=RequireArgClassMeta):
    def __init__(self, name: str, description: str, arguments: List[ToolArguments]) -> None:
        self.name: str = name
        self.description: str = description
        self.arguments: List[ToolArguments] = arguments

    class Arguments(BaseModel):
        pass

    @classmethod
    def _parse_arguments(cls) -> List[ToolArguments]:
        args = []
        arguments = cls.Arguments.model_json_schema()
        properties = arguments["properties"]
        for property_name, property_schema in properties.items():
            args.append(ToolArguments(
                name=property_name,
                description=property_schema["description"],
                required=property_name in arguments.get("required", []),
                type=property_schema["type"],
            ))
        return args

class GetWeather(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="get_weather",
            description="Returns the weather for a given location",
            arguments=GetWeather._parse_arguments()
        )
    
    class Arguments(BaseModel):
        latitude: float = Field(..., description="The latitude of the location")
        longitude: float = Field(..., description="The longitude of the location")
    
    async def __call__(self, arguments: Arguments) -> str:
        return f"Weather for {arguments.latitude} {arguments.longitude} is sunny"

class GetHoroscope(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="get_horoscope",
            description="Returns the horoscope for a given zodiac sign",
            arguments=GetHoroscope._parse_arguments()
        )
    
    class Arguments(BaseModel):
        zodiac_sign: str = Field(..., description="The zodiac sign")
    
    async def __call__(self, arguments: Arguments) -> str:
        return f"Horoscope for {arguments.zodiac_sign} is sunny"
        
    
class GetCompanyName(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="get_company_name",
            description="Returns the company name when asked",
            arguments=GetCompanyName._parse_arguments()
        )
    
    class Arguments(BaseModel):
        pass
    
    async def __call__(self, arguments: Arguments) -> str:
        return "Company name is Lowe's"