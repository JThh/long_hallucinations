def check_for_clarification_request(answer):
    """Check if answer is a clarification request."""

    list_of_clarification_requests_indicators = ['?', 'please', 'clarify',
                                                 'sorry ', 'I don\'t understand', 'I can\'t', 'there is no one']
    is_clarification_request = 0.0
    for indicator in list_of_clarification_requests_indicators:
        if indicator in answer:
            is_clarification_request = 1.0
    return is_clarification_request
