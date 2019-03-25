import re


def romanToInt(s):
    result = 0
    p1 = 'IV'
    p2 = 'IX'
    p3 = 'XL'
    p4 = 'XC'
    p5 = 'CD'
    p6 = 'CM'

    s = re.sub(p1, "!", s)
    s = re.sub(p2, "@", s)
    s = re.sub(p3, "#", s)
    s = re.sub(p4, "$", s)
    s = re.sub(p5, "%", s)
    s = re.sub(p6, "^", s)

    print('@' * 20)
    print(s)
    print('@' * 20)
    for i in s:
        if i == 'I':
            result += 1
        if i == 'v':
            result += 5
        if i == 'x':
            result += 10
        if i == 'L':
            result += 50
        if i == 'C':
            result += 100
        if i == 'D':
            result += 500
        if i =="M":
            result +=1000

        if i == '!':
            result += 4
        if i == '@':
            result += 9
        if i == '#':
            result += 40
        if i == '$':
            result += 90
        if i == '%':
            result += 400
        if i == '^':
            result += 900

    return result

s = 'MCMXCIV'
print (romanToInt(s))

