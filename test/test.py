mark = {False: ' -', True: ' Y'}
def print_table(ntypes):
  print('X' + ' '.join(ntypes))
  for row in ntypes:
    print(row, end = '')
    for col in ntypes:
        print(mark[np.can_cast(row, col)], end='')
        print()
print_table(np.typecodes['All'])