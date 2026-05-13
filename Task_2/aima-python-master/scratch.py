from logic import PropKB, expr

kb = PropKB()

# Observations
kb.tell(expr('~B11'))
kb.tell(expr('~S11'))
kb.tell(expr('B21'))
kb.tell(expr('~S21'))
kb.tell(expr('~B12'))
kb.tell(expr('S12'))

# Breeze rules
kb.tell(expr('B11 ==> P12 | P21'))
kb.tell(expr('P12 ==> B11'))
kb.tell(expr('P21 ==> B11'))

kb.tell(expr('B21 ==> P11 | P22 | P31'))
kb.tell(expr('P11 ==> B21'))
kb.tell(expr('P22 ==> B21'))
kb.tell(expr('P31 ==> B21'))

kb.tell(expr('B12 ==> P11 | P13 | P22'))
kb.tell(expr('P11 ==> B12'))
kb.tell(expr('P13 ==> B12'))
kb.tell(expr('P22 ==> B12'))

# Stench rules
kb.tell(expr('S11 ==> W12 | W21'))
kb.tell(expr('W12 ==> S11'))
kb.tell(expr('W21 ==> S11'))

kb.tell(expr('S21 ==> W11 | W22 | W31'))
kb.tell(expr('W11 ==> S21'))
kb.tell(expr('W22 ==> S21'))
kb.tell(expr('W31 ==> S21'))

kb.tell(expr('S12 ==> W11 | W13 | W22'))
kb.tell(expr('W11 ==> S12'))
kb.tell(expr('W13 ==> S12'))
kb.tell(expr('W22 ==> S12'))

# Death rules
kb.tell(expr('D11 ==> P11 | W11'))
kb.tell(expr('P11 ==> D11'))
kb.tell(expr('W11 ==> D11'))

kb.tell(expr('D22 ==> P22 | W22'))
kb.tell(expr('P22 ==> D22'))
kb.tell(expr('W22 ==> D22'))

print("Query P31:", kb.ask(expr('P31')))
print("Query W13:", kb.ask(expr('W13')))
