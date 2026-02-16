"""Portfolio-specific tools for the ai_server application."""

from pydantic import BaseModel, Field
from typing import Any

from omniagent.ai.tools.tools import Tool


class GetCompanyName(Tool):
    """Tool to get the company name - no arguments required."""
    
    class Arguments(BaseModel):
        pass
    
    def __init__(self) -> None:
        super().__init__(
            name="get_company_name",
            description="Get the company name"
        )
    
    async def __call__(self, arguments: Arguments) -> Any:
        # Mock implementation - returns a fixed company name
        return "Acme Corporation"


class GetHoroscope(Tool):
    """Tool to get the horoscope for a given zodiac sign."""
    
    class Arguments(BaseModel):
        sign: str = Field(..., description="The zodiac sign (e.g., 'aries', 'taurus', etc.)")
    
    def __init__(self) -> None:
        super().__init__(
            name="get_horoscope",
            description="Get the daily horoscope for a given zodiac sign"
        )
    
    async def __call__(self, arguments: Arguments) -> Any:
        # Mock implementation - replace with actual API call
        horoscopes = {
            "aries": "Today is a great day for new beginnings. Take initiative!",
            "taurus": "Focus on stability and comfort today. Good things come to those who wait.",
            "gemini": "Communication is key today. Express yourself clearly.",
            "cancer": "Trust your intuition today. Your emotions guide you well.",
            "leo": "Your creativity shines today. Share your talents with others.",
            "virgo": "Pay attention to details today. Organization brings success.",
            "libra": "Seek balance in all things today. Harmony is within reach.",
            "scorpio": "Transformation is in the air. Embrace change.",
            "sagittarius": "Adventure calls today. Explore new horizons.",
            "capricorn": "Hard work pays off today. Stay focused on your goals.",
            "aquarius": "Innovation is your strength today. Think outside the box.",
            "pisces": "Dreams hold meaning today. Trust your imagination.",
        }
        sign_lower = arguments.sign.lower()
        return horoscopes.get(sign_lower, f"Unknown zodiac sign: {arguments.sign}")
