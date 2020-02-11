# Based on https://stackoverflow.com/questions/42590512/how-to-convert-from-infix-to-postfix-prefix-using-ast-python-module/42591495

import ast

class Visitor(ast.NodeVisitor):
    """"""

    def __init__(self):
        """Constructor"""

        self.tokens = []

    def f_continue(self, node):
        """"""

        super(Visitor, self).generic_visit(node)

    def visit_Add(self, node):
        """"""

        self.tokens.append('+')
        self.f_continue(node)

    def visit_Sub(self, node):
        """"""

        self.tokens.append('-')
        self.f_continue(node)

    def visit_And(self, node):
        """"""

        self.tokens.append('&&')
        self.f_continue(node)

    def visit_BinOp(self, node):
        """"""

        pass

    def visit_BoolOp(self, node):
        """"""

        pass

    def visit_Call(self, node):
        """"""

        pass

    def visit_Div(self, node):
        """"""

        self.tokens.append('/')
        self.f_continue(node)

    def visit_Expr(self, node):
        """"""
        self.f_continue(node)

    def visit_Import(self, stmt_import):
        """"""

        for alias in stmt_import.names:
            print(f'import name "{alias.name}"')
            print(f'import object {alias}')
        self.f_continue(stmt_import)

    def visit_Load(self, node):
        """"""

        self.f_continue(node)

    def visit_Module(self, node):
        """"""

        self.f_continue(node)

    def visit_Mult(self, node):
        """"""

        self.tokens.append('*')
        self.f_continue(node)

    def visit_Name(self, node):
        """"""

        self.tokens.append(node.id)
        self.f_continue(node)

    def visit_NameConstant(self, node):
        """"""

        self.tokens.append(node.value)
        self.f_continue(node)

    def visit_Num(self, node):
        """"""

        self.tokens.append(node.n)
        self.f_continue(node)

    def visit_Pow(self, node):
        """"""

        self.tokens.append('pow')
        self.f_continue(node)

class PostfixVisitor(Visitor):
    """ Visitor Class for Reverse Polish Notation """

    def __init__(self):
        """Constructor"""

        super(PostfixVisitor, self).__init__()

    def visit_BinOp(self, node):
        """"""

        self.visit(node.left)
        self.visit(node.right)
        self.visit(node.op)

    def visit_BoolOp(self, node):
        """"""

        for val in node.values:
            self.visit(val)
        self.visit(node.op)

    def visit_Call(self, node):
        """"""

        for arg in node.args:
            self.visit(arg)
        self.visit(node.func)

class PrefixVisitor(Visitor):
    """ Visitor Class for Polish Notation """

    def __init__(self):
        """Constructor"""

        super(PrefixVisitor, self).__init__()

    def visit_BinOp(self, node):
        """"""

        self.visit(node.op)
        self.visit(node.left)
        self.visit(node.right)

    def visit_BoolOp(self, node):
        """"""

        self.visit(node.op)
        for val in node.values:
            self.visit(val)

    def visit_Call(self, node):
        """"""

        self.visit(node.func)
        for arg in node.args:
            self.visit(arg)

def convert_infix_to_rpn(infix_expression, temp_repl=None, post_repl=None,
                         double_format='.12g'):
    """"""

    mod_infix = infix_expression
    if temp_repl is not None:
        for _original, _temp in temp_repl:
            mod_infix = mod_infix.replace(_original, _temp)

    visitor = PostfixVisitor()
    visitor.visit(ast.parse(mod_infix))

    rpn_expr = ' '.join([f'{v:{double_format}}' if isinstance(v, float) else f'{v}'
                         for v in visitor.tokens])
    if temp_repl is not None:
        for _original, _temp in temp_repl:
            rpn_expr = rpn_expr.replace(_temp, _original)
    if post_repl is not None:
        for current, new in post_repl:
            rpn_expr = rpn_expr.replace(current, new)

    return rpn_expr

class RPN:
    """"""

    def __init__(self, rpn_epxression):
        """Constructor"""

        self.rpn_expr = rpn_epxression

        self.buffer = []

        self.operator_list = ['+', '-', '*', '/', '**', '&&', '||']

        self.func_list_1arg = [
            'ln', 'exp', 'ceil', 'floor', 'int', 'sin', 'cos', 'tan',
            'asin', 'acos', 'atan', 'dsin', 'dcos', 'dtan',
            'dasin', 'dacos', 'datan', 'sinh', 'cosh', 'tanh',
            'asinh', 'acosh', 'atanh', 'sqr', 'sqrt', 'abs',
            'chs', 'rec', 'rtod', 'dtor',
        ]
        self.func_list_2args = ['pow', 'atan2', 'hypot', 'max2', 'min2',]
        self.func_list_3args = ['segt', 'selt', 'sene',]
        self.func_list_special = ['maxn', 'minn']

        self.func_list = self.func_list_1arg + self.func_list_2args
        self.func_list += self.func_list_3args
        self.func_list += self.func_list_special

    def clear_buffer(self):
        """"""

        self.buffer.clear()

    def _operate(self, op_name):
        """"""

        v2 = self.buffer.pop()
        v1 = self.buffer.pop()

        if len(v1.split()) != 1:
            v1 = f'({v1})'
        if len(v2.split()) != 1:
            v2 = f'({v2})'

        if op_name == '&&':
            op_name = 'and'
        elif op_name == '||':
            op_name = 'or'

        return f'{v1} {op_name} {v2}'

    def toinfix(self):
        """"""

        token_list = self.rpn_expr.split()

        for token in token_list:

            if token in self.func_list:
                #self.buffer.append(getattr(self, token)())
                if token in self.func_list_1arg:
                    v1 = self.buffer.pop()
                    self.buffer.append(f'{token}({v1})')
                elif token in self.func_list_2args:
                    v2 = self.buffer.pop()
                    v1 = self.buffer.pop()
                    self.buffer.append(f'{token}({v1}, {v2})')
                elif token in self.func_list_3args:
                    v3 = self.buffer.pop()
                    v2 = self.buffer.pop()
                    v1 = self.buffer.pop()
                    self.buffer.append(f'{token}({v1}, {v2}, {v3})')
                elif token in self.func_list_special:
                    if token in ('maxn', 'minn'):
                        n = int(self.buffer.pop())
                        _args = [self.buffer.pop() for _ in range(n)]
                        _arg_str = ', '.join(_args[::-1])
                        self.buffer.append(f'{token}({_arg_str})')
                    else:
                        raise ValueError('Special function "{token}" not defined')
                else:
                    raise ValueError('You should not see this error at all')
            elif token in self.operator_list:
                self.buffer.append(self._operate(token))
            else:
                self.buffer.append(token)

        assert len(self.buffer) == 1

        return self.buffer[0]

def convert_rpn_to_infix(rpn_expression):
    """"""

    r = RPN(rpn_expression)

    return r.toinfix()

if __name__ == '__main__':

    formulas = [
        "1+2",
        "1+2*3",
        "1/2",
        "(1+2)*3",
        "sin(x)*x**2",
        "cos(x)",
        "True and False",
        "sin(w*time)",
        '(5 + 3) * ((3 **2 - 1) + sin(cos(x)))',
        'abs(dnux_dp * 15)',
        'abs(dnux_dp * 15.3)',
    ]

    if False:
        node = ast.parse('(5 + 3) * ((3 **2 - 1) + sin(cos(x)))')
        print(ast.dump(node))


    print('** Postfix or RPN (Reverse Polish Notation) **\n')

    for index, f in enumerate(formulas):
        print(f'{index} - {f:*^76}')
        visitor = PostfixVisitor()
        visitor.visit(ast.parse(f))
        print(visitor.tokens)
        print(f'Re-convereted to Infix: {convert_rpn_to_infix(convert_infix_to_rpn(f))}')
        print(' ')

    print('** Prefix or PN (Polish Notation) **\n')

    for index, f in enumerate(formulas):
        print(f'{index} - {f:*^76}')
        visitor = PrefixVisitor()
        visitor.visit(ast.parse(f))
        print(visitor.tokens)
        print(' ')