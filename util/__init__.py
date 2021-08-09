from hanspell import spell_checker


def correct_sent(sent):
    spelled_sent = spell_checker.check(sent)
    return spelled_sent.checked
