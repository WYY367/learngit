import sys
sys.path.insert(0, '.')
from src.chains.prompts import format_similar_defects, format_value

# Test format_value
print('Testing format_value:')
print(f'  format_value(None) = "{format_value(None)}"')
print(f'  format_value("") = "{format_value("")}"')
print(f'  format_value("N/A") = "{format_value("N/A")}"')
print(f'  format_value("Test Summary") = "{format_value("Test Summary")}"')

# Test format_similar_defects
print('\nTesting format_similar_defects:')
test_defects = [
    {
        'metadata': {
            'Identifier': '12345',
            'Summary': 'Unused variable found in ASW component',
            'Component': 'ASW-PR',
            'Customer': 'GAMC',
            'CategoryOfGaps': 'Imp: Practise',
            'SubCategoryOfGaps': 'Others'
        },
        'text': 'Impact: Base variable not used...',
        'score': 0.92
    },
    {
        'metadata': {
            'Identifier': '67890',
            'Summary': '',  # Empty summary
            'Component': None,  # None value
        },
        'text': 'Some description',
        'score': 0.85
    }
]

result = format_similar_defects(test_defects, 'zh')
print(result[:800])
