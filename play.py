s=''
if len(s)==0:
    print ('fadfdasf')
stack = []
stack.append(s[0])
s = s[1:]
for i in s:
            if i == ')' and  ord(i)-ord(stack[-1])==1:
                stack.pop(-1)
            elif i == ']' and ord(i)-ord(stack[-1])==2:
                stack.pop(-1)
            elif i =='}' and ord(i)-ord(stack[-1])==2:
                stack.pop(-1)
            else:
                stack.append(i)
pass