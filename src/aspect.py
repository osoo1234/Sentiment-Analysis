import re

class AspectExtractor:
    def __init__(self, language='en'):
        self.language = language
        if language == 'en':
            self.aspect_keywords = {
                'Customer Service': ['service', 'staff', 'attendant', 'crew', 'support', 'help', 'call', 'agent', 'phone'],
                'Flight Experience': ['flight', 'delay', 'time', 'cancelled', 'plane', 'seat', 'late', 'schedule', 'status'],
                'Pricing': ['price', 'cost', 'ticket', 'fare', 'cheap', 'expensive', 'money', 'charge', 'refund'],
                'Baggage': ['bag', 'luggage', 'suitcase', 'checked', 'lost']
            }
        elif language == 'ar':
            # Arabic aspects (basic)
            self.aspect_keywords = {
                'خدمة العملاء': ['خدمه', 'موظف', 'دعم', 'مساعده', 'اتصال', 'رد', 'تواصل'],
                'تجربة الرحلة': ['رحلة', 'تاخير', 'وقت', 'الغاء', 'طياره', 'مقعد', 'تاخر', 'جدول'],
                'الأسعار': ['سعر', 'تكلفه', 'تذكره', 'رخيص', 'غالي', 'فلوس', 'دفع', 'استرداد'],
                'الأمتعة': ['حقيبه', 'شنطه', 'امتعه', 'فقدان', 'ضياع']
            }

    def detect_aspect(self, text):
        if not isinstance(text, str):
            return "General"
        
        text = text.lower()
        counts = {aspect: 0 for aspect in self.aspect_keywords}
        
        for aspect, keywords in self.aspect_keywords.items():
            for kw in keywords:
                if re.search(r'\b' + kw + r'\b', text):
                    counts[aspect] += 1
                    
        max_aspect = max(counts, key=counts.get)
        if counts[max_aspect] == 0:
            return "General"
        return max_aspect
