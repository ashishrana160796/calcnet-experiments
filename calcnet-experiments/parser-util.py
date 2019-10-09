# A shunting yard based algorithm variant for parsing both left and right associative operators.

def precedence_order(op):
    if op == '+' or op == '-': 
        return 1
    elif op == '*' or op == '/': 
        return 2
    # Here, '_' represents square-root function
    elif op == '^' or op == '_' : 
        return 3
    return 0

import math

def apply_op(a, b, op):
    if op == '+': return a + b
    if op == '-': return a - b
    if op == '*': return a * b
    if op == '/': return a / b 
    if op == '^': return math.pow(a,b)
    if op == '_': return math.sqrt(a,b)

def eval_ops(tokens): 
    # equivalent stack structure to store integer values. 
    values = [] 
    # equivalent stack structure stack to store operators. 
    ops = [] 
    i = 0
    while i < len(tokens): 
        # skipping blank tokens.
        if tokens[i] == ' ': 
            i += 1
            continue
        # pushing opening brace to 'ops' stack.
        elif tokens[i] == '(': 
            ops.append(tokens[i]) 
        # Push number tokens into stack.
        elif tokens[i].isdigit():
            val = 0
			# Multiple digit token parsing.
            while (i < len(tokens) and tokens[i].isdigit()): 
                val = (val * 10) + int(tokens[i]) 
                i += 1
            values.append(val) 
        # Evaluating sub-expressions on closing brace
        elif tokens[i] == ')':
            while len(ops) != 0 and ops[-1] != '(': 
                val2 = values.pop() 
                val1 = values.pop() 
                op = ops.pop() 
                values.append(apply_op(val1, val2, op)) 
            # pop off the opening brace. 
            ops.pop() 
        # Current token is an operator. 
        else: 
            # top of 'ops' has same or greater precedence to current token, which is an operator. 
            # Apply operator on top of 'ops' and to top two elements in values stack if left associative operator.
            # otherwise push the operator into the stack.
            while (len(ops) != 0 and
                precedence_order(ops[-1]) >= precedence_order(tokens[i]) and
                # handling right-associative exponential, squaring or square root operators.
                not (precedence_order(ops[-1]) == 3 and precedence_order(tokens[i])==3) ):		
                val2 = values.pop() 
                val1 = values.pop() 
                op = ops.pop() 
                values.append(apply_op(val1, val2, op))
            # Push current lower precidence or right associative operator token to 'ops'. 
            ops.append(tokens[i]) 
        i += 1
    # Parsing the remaining expression
    while len(ops) != 0:	
        val2 = values.pop() 
        val1 = values.pop() 
        op = ops.pop() 		
        values.append(apply_op(val1, val2, op)) 
	# Return the final remaining result.
    return values[-1]

if __name__ == "__main__":
    print("Result of 2 ^ 3 ^ 2")
    print(eval_ops("2 ^ 3 ^ 2"))
    print("Result of 100 * 2 + 47")
    print(eval_ops("100 * 2 + 47"))
    print("Result of 100 * ( 2 - 12 )")
    print(eval_ops("100 * ( 2 - 12 )"))
    print("Result of 100 * ( 2 + 12 ) / 11")
    print(eval_ops("100 * ( 2 + 12 ) / 11"))
