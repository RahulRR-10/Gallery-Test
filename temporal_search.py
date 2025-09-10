"""
Temporal Intelligence Module
Phase 2 Stage 1: Time-based photo search functionality
"""

import dateparser
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any
import re

class TemporalParser:
    """Parse natural language time expressions into Unix timestamps"""
    
    def __init__(self):
        """Initialize the temporal parser"""
        self.current_time = datetime.now()
        
    def parse_time_expression(self, time_expr: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Parse natural language time expression into start/end timestamps
        
        Args:
            time_expr: Natural language time expression like "last year", "2020", "last Christmas"
            
        Returns:
            Tuple of (start_timestamp, end_timestamp) or (None, None) if parsing fails
        """
        if not time_expr:
            return None, None
            
        time_expr = time_expr.lower().strip()
        
        # Handle specific year (e.g., "2020", "2023")
        year_match = re.match(r'^(\d{4})$', time_expr)
        if year_match:
            year = int(year_match.group(1))
            return self._get_year_range(year)
        
        # Handle year ranges (e.g., "2018-2022", "2020 to 2023")
        year_range_match = re.match(r'^(\d{4})[-\s]*(?:to|-|\s)+\s*(\d{4})$', time_expr)
        if year_range_match:
            start_year = int(year_range_match.group(1))
            end_year = int(year_range_match.group(2))
            start_ts = self._get_year_range(start_year)[0]
            end_ts = self._get_year_range(end_year)[1]
            return start_ts, end_ts
        
        # Handle relative expressions with dateparser
        try:
            # Special handling for common expressions
            if 'last christmas' in time_expr:
                return self._get_last_christmas_range()
            elif 'this christmas' in time_expr:
                return self._get_this_christmas_range()
            elif 'last year' in time_expr:
                return self._get_last_year_range()
            elif 'this year' in time_expr:
                return self._get_this_year_range()
            elif 'last month' in time_expr:
                return self._get_last_month_range()
            elif 'this month' in time_expr:
                return self._get_this_month_range()
            elif 'last week' in time_expr:
                return self._get_last_week_range()
            elif 'this week' in time_expr:
                return self._get_this_week_range()
            elif 'yesterday' in time_expr:
                return self._get_yesterday_range()
            elif 'today' in time_expr:
                return self._get_today_range()
            
            # Try dateparser for other expressions
            parsed_date = dateparser.parse(time_expr)
            if parsed_date:
                # For single dates, create a day range
                start_of_day = parsed_date.replace(hour=0, minute=0, second=0, microsecond=0)
                end_of_day = parsed_date.replace(hour=23, minute=59, second=59, microsecond=999999)
                return int(start_of_day.timestamp()), int(end_of_day.timestamp())
                
        except Exception as e:
            print(f"⚠️ Error parsing time expression '{time_expr}': {e}")
            
        return None, None
    
    def _get_year_range(self, year: int) -> Tuple[int, int]:
        """Get start and end timestamps for a specific year"""
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31, 23, 59, 59, 999999)
        return int(start.timestamp()), int(end.timestamp())
    
    def _get_last_christmas_range(self) -> Tuple[int, int]:
        """Get timestamp range for last Christmas (Dec 24-26)"""
        current_year = self.current_time.year
        last_year = current_year - 1 if self.current_time.month < 12 else current_year
        
        start = datetime(last_year, 12, 24)
        end = datetime(last_year, 12, 26, 23, 59, 59, 999999)
        return int(start.timestamp()), int(end.timestamp())
    
    def _get_this_christmas_range(self) -> Tuple[int, int]:
        """Get timestamp range for this Christmas (Dec 24-26)"""
        current_year = self.current_time.year
        start = datetime(current_year, 12, 24)
        end = datetime(current_year, 12, 26, 23, 59, 59, 999999)
        return int(start.timestamp()), int(end.timestamp())
    
    def _get_last_year_range(self) -> Tuple[int, int]:
        """Get timestamp range for last year"""
        last_year = self.current_time.year - 1
        return self._get_year_range(last_year)
    
    def _get_this_year_range(self) -> Tuple[int, int]:
        """Get timestamp range for this year"""
        current_year = self.current_time.year
        return self._get_year_range(current_year)
    
    def _get_last_month_range(self) -> Tuple[int, int]:
        """Get timestamp range for last month"""
        # Get first day of current month, then subtract one day to get last month
        first_day_current = self.current_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        last_day_last_month = first_day_current - timedelta(days=1)
        first_day_last_month = last_day_last_month.replace(day=1)
        
        return int(first_day_last_month.timestamp()), int(last_day_last_month.timestamp())
    
    def _get_this_month_range(self) -> Tuple[int, int]:
        """Get timestamp range for this month"""
        first_day = self.current_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Get last day of current month
        if self.current_time.month == 12:
            last_day = datetime(self.current_time.year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = datetime(self.current_time.year, self.current_time.month + 1, 1) - timedelta(days=1)
        
        last_day = last_day.replace(hour=23, minute=59, second=59, microsecond=999999)
        return int(first_day.timestamp()), int(last_day.timestamp())
    
    def _get_last_week_range(self) -> Tuple[int, int]:
        """Get timestamp range for last week (Monday to Sunday)"""
        days_since_monday = self.current_time.weekday()
        start_of_this_week = self.current_time - timedelta(days=days_since_monday)
        start_of_this_week = start_of_this_week.replace(hour=0, minute=0, second=0, microsecond=0)
        
        start_of_last_week = start_of_this_week - timedelta(days=7)
        end_of_last_week = start_of_this_week - timedelta(seconds=1)
        
        return int(start_of_last_week.timestamp()), int(end_of_last_week.timestamp())
    
    def _get_this_week_range(self) -> Tuple[int, int]:
        """Get timestamp range for this week (Monday to Sunday)"""
        days_since_monday = self.current_time.weekday()
        start_of_week = self.current_time - timedelta(days=days_since_monday)
        start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
        
        end_of_week = start_of_week + timedelta(days=6, hours=23, minutes=59, seconds=59, microseconds=999999)
        
        return int(start_of_week.timestamp()), int(end_of_week.timestamp())
    
    def _get_yesterday_range(self) -> Tuple[int, int]:
        """Get timestamp range for yesterday"""
        yesterday = self.current_time - timedelta(days=1)
        start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        end = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
        return int(start.timestamp()), int(end.timestamp())
    
    def _get_today_range(self) -> Tuple[int, int]:
        """Get timestamp range for today"""
        start = self.current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        end = self.current_time.replace(hour=23, minute=59, second=59, microsecond=999999)
        return int(start.timestamp()), int(end.timestamp())
    
    def format_timestamp_range(self, start_ts: Optional[int], end_ts: Optional[int]) -> str:
        """
        Format timestamp range for display
        
        Args:
            start_ts: Start timestamp
            end_ts: End timestamp
            
        Returns:
            Formatted string representation
        """
        if start_ts is None and end_ts is None:
            return "No time filter"
        elif start_ts is None and end_ts is not None:
            return f"Before {datetime.fromtimestamp(end_ts).strftime('%Y-%m-%d')}"
        elif end_ts is None and start_ts is not None:
            return f"After {datetime.fromtimestamp(start_ts).strftime('%Y-%m-%d')}"
        elif start_ts is not None and end_ts is not None:
            start_date = datetime.fromtimestamp(start_ts).strftime('%Y-%m-%d')
            end_date = datetime.fromtimestamp(end_ts).strftime('%Y-%m-%d')
            if start_date == end_date:
                return f"On {start_date}"
            else:
                return f"From {start_date} to {end_date}"
        else:
            return "Invalid time range"


def test_temporal_parser():
    """Test the temporal parser with various expressions"""
    parser = TemporalParser()
    
    test_expressions = [
        "2020",
        "2018-2022",
        "last year",
        "this year", 
        "last month",
        "this month",
        "last week",
        "this week",
        "yesterday",
        "today",
        "last Christmas",
        "this Christmas"
    ]
    
    print("🕒 Testing Temporal Parser:")
    print("=" * 50)
    
    for expr in test_expressions:
        start_ts, end_ts = parser.parse_time_expression(expr)
        formatted = parser.format_timestamp_range(start_ts, end_ts)
        print(f"'{expr}' -> {formatted}")
    
    print("=" * 50)
    print("✅ Temporal parser test completed!")


if __name__ == "__main__":
    test_temporal_parser()
