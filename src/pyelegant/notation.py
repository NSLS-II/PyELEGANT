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

def convert_infix_to_rpn(infix_expression, temp_repl=None):
    """"""

    mod_infix = infix_expression
    if temp_repl is not None:
        for _original, _temp in temp_repl:
            mod_infix = mod_infix.replace(_original, _temp)

    visitor = PostfixVisitor()
    visitor.visit(ast.parse(mod_infix))

    rpn_expr = ' '.join([str(v) for v in visitor.tokens])
    if temp_repl is not None:
        for _original, _temp in temp_repl:
            rpn_expr = rpn_expr.replace(_temp, _original)

    return rpn_expr

def convert_rpn_to_infix(rpn_expression):
    """"""

    raise NotImplementedError()

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
        print(' ')

    print('** Prefix or PN (Polish Notation) **\n')

    for index, f in enumerate(formulas):
        print(f'{index} - {f:*^76}')
        visitor = PrefixVisitor()
        visitor.visit(ast.parse(f))
        print(visitor.tokens)
        print(' ')