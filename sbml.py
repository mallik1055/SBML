#Mallikarjuna Rao Budida(112503844)
import sys
from random import randint

class SemanticError(Exception):
    pass

primitiveNodeNames = ('StringNode','NumberNode','BoolNode','TupleNode','ListNode')

#Primitive Data Type Nodes
class Node:
    def __init__(self):
        pass

    def reduce(self):
        return 0

    def is_primitive(self):
        #print(self.__class__.__name__)
        if self.__class__.__name__  in primitiveNodeNames:
            return 1
        else:
            return 0

    def evaluate(self): #evaluates to the bare bone primitive type
        if not self.is_primitive():
            return self.reduce().reduce()
        else:
            return self.reduce()

    def reduceToPrimNode(self):
        if not self.is_primitive():
            x = self.reduce()
            return x

        else:
            return self

    def __eq__(self, other):
        return self.value == other.value


class NumberNode(Node):
    def __init__(self, v):
        v = str(v)
        if ('.' in v):
            self.value = float(v)
        else:
            self.value = int(v)

    def reduce(self):
        return self.value


class StringNode(Node):
    def __init__(self,v,strip=0):
        if strip:
            self.value = v[1:-1] #strip out start and end quotes
        else:
            self.value = v
    def reduce(self):
        return self.value


class BoolNode(Node):
    def __init__(self,v):
        v = str(v)
        if v == 'True':
            self.value = True
        else:
            self.value = False

    def reduce(self):
        return self.value


class ListNode(Node):
    def __init__(self,v):
        self.value = v

    def reduce(self):
        for i in range(len(self.value)) :
            self.value[i] = self.value[i].reduceToPrimNode()
        return self.value

    def serialize(self):
        serializedList = []
        for e in self.evaluate() :
            e = e.reduceToPrimNode()

            if e.__class__.__name__ in ('ListNode','TupleNode'):
                e_val = e.serialize()
            else:
                e_val = e.evaluate()
            
            serializedList+=[e_val]
        return serializedList

class TupleNode(Node):
    def __init__(self,v):
        self.value = v
    def reduce(self):
        return self.value
    
    def serialize(self):
        serializedTuple = ()
        for e in self.evaluate() :
            
            if e.__class__.__name__ == 'ListNode':
                e_val = e.serialize()    
            elif e.__class__.__name__ == 'TupleNode':
                e_val = e.serialize()
            else:
                e_val = e.evaluate()
            
            serializedTuple+=(e_val,)
        return serializedTuple

class VarNode(Node):
    def __init__(self, v):
        self.name = str(v)
        self.queryIndexList = []

    def pushQueryIndex(self,index):
        self.queryIndexList += [index]

    def reduce(self):

        target,target_key = self.getTarget()
        return target[target_key]
 
    def getName(self):
        return self.name


    def getTarget(self):
        target_key_ = None
        target_ = None
        
        if len(self.queryIndexList) is 0 :
            target_ = varNameValueIndex
            target_key_ = self.name 
        else:
            i = 0
            target_ = varNameValueIndex[self.name] #listNode
            temp_target_ = target_
            while(i < len(self.queryIndexList) - 1 ):
                temp_target_ = temp_target_.reduce() #List
                temp_target_ = temp_target_[self.queryIndexList[i].reduce()] #listNode
                i+=1
            target_ = temp_target_.reduce() #List
            target_key_ = self.queryIndexList[-1].evaluate()
        
        return target_,target_key_
 
# Operation Nodes


class varInitOpNode(Node):
    def __init__(self,varNode,varValueNode):
        self.varNode = varNode
        self.varValueNode = varValueNode
    
    def evaluate(self):

        varValueNode = self.varValueNode.reduceToPrimNode()

        target,target_key = self.varNode.getTarget()
        target[target_key] = varValueNode
        # self.varNode.reduce()


class PrintNode(Node):
    def __init__(self, v):
        self.value = v

    def evaluate(self):

        self_value = self.value.reduceToPrimNode()

        self_class_name = self_value.__class__.__name__ 
        
        if self_class_name == 'StringNode':
            #print("'"+self_value.reduce()+"'")
            print(self_value.reduce())

        elif self_class_name == 'ListNode':
            auxList = self_value.serialize()
            print(auxList)

        elif self_class_name == 'TupleNode':
            auxTuple = self_value.serialize()
            print(auxTuple)


        elif self_class_name == 'BoolNode':
            if(self_value.reduce()):
                print('True')
            else:
                print('False')
        elif self_class_name == 'NumberNode':
            print(self_value.reduce())
        else:
            #Should be here
            #for safety
            raise SemanticError()
            #print(self.value.reduce())

class BinopNode(Node):
    def __init__(self, op, v1, v2):
        self.v1 = v1
        self.v2 = v2
        self.op = op

    def reduce(self):

        v1 = self.v1.reduceToPrimNode()
        v2 = self.v2.reduceToPrimNode()

        #comparisions should allow only strings & Numbers
        if isinstance(v1.reduce(), bool) or isinstance(v2.reduce(),bool):
            is_bool_input = 1
        else:
            is_bool_input = 0

        #we compare nodes here rather than primitive data types
        #to avoid complexity of integer,float comparision
        if type(v1) == type(v2):
            is_homog_operands = 1
        else:
            is_homog_operands = 0

        if not is_homog_operands:
            raise SemanticError("Operators in BinOp are not homog")

        operandNodeType = v1.__class__.__name__  #Can be among ['NumberNode','StringNode','BoolNode','ListNode']

        availGroups = {
            '1':['ListNode','NumberNode','StringNode'],
            '2':['NumberNode'],
            '3':['NumberNode','StringNode'],
            '4':['BoolNode']
        }

        operandNodeGroup = []
        for k,v in availGroups.items():
            if operandNodeType in v:
                operandNodeGroup+=[int(k)]


        #all operate on Homog operand types

        #lists, number, string
        #grp 1
        if (self.op == '+'):
            if 1 not in operandNodeGroup:
                raise SemanticError("NodeGrp mismatch")

            retval = v1.evaluate() + v2.evaluate()
            
            if isinstance(retval, str):
                retval = StringNode(retval)
            elif isinstance(retval, list):
                retval = ListNode(retval)
            else:
                retval = NumberNode(retval)
        #numbers
        #grp 2
        elif (self.op == '-'):
            if 2 not in operandNodeGroup:
                raise SemanticError("NodeGrp mismatch")
            retval = NumberNode(v1.reduce() - v2.reduce())
        elif (self.op == '*'):
            if 2 not in operandNodeGroup:
                raise SemanticError("NodeGrp mismatch")
            retval = NumberNode(v1.reduce() * v2.reduce())
        elif (self.op == '/'):
            if 2 not in operandNodeGroup:
                raise SemanticError("NodeGrp mismatch")
            retval = NumberNode(v1.reduce() / v2.reduce())
        elif (self.op == '**'):
            if 2 not in operandNodeGroup:
                raise SemanticError("NodeGrp mismatch")
            retval = NumberNode(v1.reduce() ** v2.reduce())
        elif(self.op == 'div'):
            if 2 not in operandNodeGroup:
                raise SemanticError("NodeGrp mismatch")
            retval = NumberNode(v1.reduce() // v2.reduce())
        elif(self.op == 'mod'):
            if 2 not in operandNodeGroup:
                raise SemanticError("NodeGrp mismatch")
            retval = NumberNode(v1.reduce() % v2.reduce())

        #Numbers and strings
        #grp 3
        elif(self.op == '<'):
            if 3 not in operandNodeGroup:
                raise SemanticError("NodeGrp mismatch")
            retval = BoolNode( v1.reduce() < v2.reduce())
        elif(self.op == '<='):
            if 3 not in operandNodeGroup:
                raise SemanticError("NodeGrp mismatch")
            retval = BoolNode(v1.reduce() <= v2.reduce())
        elif(self.op == '>'):
            if 3 not in operandNodeGroup:
                raise SemanticError("NodeGrp mismatch")
            retval = BoolNode(v1.reduce() > v2.reduce())
        elif(self.op == '>='):
            if 3 not in operandNodeGroup:
                raise SemanticError("NodeGrp mismatch")
            retval = BoolNode(v1.reduce() >= v2.reduce())
        elif (self.op == '=='):
            if 3 not in operandNodeGroup:
                raise SemanticError("NodeGrp mismatch")
            retval =  BoolNode(v1.reduce() == v2.reduce())
        elif(self.op == '<>'):
            if 3 not in operandNodeGroup:
                raise SemanticError("NodeGrp mismatch")
            retval = BoolNode(v1.reduce() != v2.reduce())
        #Boolean
        #grp4
        elif(self.op == 'orelse'):
            if 4 not in operandNodeGroup:
                raise SemanticError("NodeGrp mismatch")
            retval = BoolNode(v1.reduce() or v2.reduce())
        elif(self.op == 'andalso'):
            if 4 not in operandNodeGroup:
                raise SemanticError("NodeGrp mismatch")
            retval = BoolNode(v1.reduce() and v2.reduce())

        return retval


class InListOpNode(Node):
    def __init__(self,n,h):#needle,haystack
        self.n = n
        self.h = h
    def reduce(self):
        self.value = BoolNode('False')

        if isinstance(self.h.evaluate(),str):
            if self.n.evaluate() in self.h.evaluate():
                self.value = BoolNode('True')
        else:
            for i in self.h.evaluate():
                if i.evaluate() == self.n.evaluate():
                    self.value = BoolNode('True')

        return self.value

class ListItemOpNode(Node):
    def __init__(self,l,i): #list #index
        self.l = l
        self.i = i
    def reduce(self):
        #list can be a string or normal list
        if isinstance(self.l.evaluate(),str):
            return StringNode((self.l.evaluate())[self.i.evaluate()])
        elif isinstance(self.l.evaluate(),list):
            k = self.i.evaluate()
            return ((self.l.evaluate())[self.i.evaluate()]).reduceToPrimNode()
        else:
            raise SemanticError()

class BlockNode(Node):
    def __init__(self,sl):
        self.value = sl

    def evaluate(self):
        for statement in self.value:
            statement.evaluate()

class WhileNode(Node):
    def __init__(self,cond,block):
        self.cond = cond
        self.block = block
    def evaluate(self):

        while(self.cond.evaluate()):
            self.block.evaluate()


class IfElseNode(Node):
    def __init__(self,cond,block_1,block_2):
        self.cond = cond
        self.block_1 = block_1
        self.block_2 = block_2
    
    def evaluate(self):
        # print(self.cond.reduce())


        cond = self.cond.evaluate()

        if(cond):
            self.block_1.evaluate()
        elif self.block_2 is not None :
            self.block_2.evaluate()
        else:
            pass

class NotOpNode(Node):
    def __init__(self,v):
        self.value = v
    def reduce(self):

        value = self.value
        return BoolNode(not value.evaluate())

class appendTupleOp(Node):
    def __init__(self,t,v):#tup1,val to append
        self.t = t
        self.v = v
    def reduce(self):
        return TupleNode( self.t.evaluate() + (self.v.reduceToPrimNode(),))

class appendListOpNode(Node):
    def __init__(self,e,L): #element_to_append,List
        self.e = e
        self.L = L
    def reduce(self):
        return ListNode(self.L.evaluate() + [self.e.reduceToPrimNode()])

class hashOpNode(Node):
    def __init__(self,n,h):#needle,haystack
        self.n = n
        self.h = h
    def reduce(self):
        return self.h.evaluate()[self.n.evaluate()-1]

class ConsOpNode(Node):
    def __init__(self,i,L):#item_to_cons,List
        self.i = i
        self.L = L

    def reduce(self):
        return ListNode([self.i.reduceToPrimNode()]+self.L.evaluate())


reserved = {
    'if'     : 'IF',
    'else'   : 'ELSE',
    'while'  : 'WHILE',
    'print'  : 'PRINT',
    'in'     : 'IN',
    'div'    : 'QDIV',
    'mod'    : 'MOD',
    'andalso': 'AND',
    'not'    : 'NOT',
    'orelse' : 'OR',
    'True'   : 'TRUE',
    'False'  : 'FALSE',
    'fun'    : 'FUN'
}

# TODOLIST
# def t_TRUE(t):
#     'True|true'
#     t.value = BoolNode('True')
#     return t

# def t_FALSE(t):
#     'False|false'
#     t.value = BoolNode('False')
#     return t



tokens = [
    'LPAREN', 'RPAREN',
    'NAME','NUMBER', 'STRING',
    'PLUS', 'MINUS', 'TIMES', 'DIVIDE','POWER',
    'LBRACKET', 'RBRACKET',
    'LCURLY','RCURLY',
    'CONS','EQUALS',
    'LESSERTHAN','LESSEQ','GREATERTHAN','GREATEREQ',
    'DEQUAL','NOTEQUAL',
    'HASH',
    'ID'
]

tokens = tokens + list(reserved.values())




# Parsing rules
#From http://www.mathcs.emory.edu/~valerie/courses/fall10/155/resources/op_precedence.html
precedence = (
    ('right','EQUALS'),
    ('left','OR'),
    ('left','AND'),
    ('left','NOT'),
    ('left','DEQUAL','NOTEQUAL','LESSERTHAN','LESSEQ','GREATERTHAN','GREATEREQ','IN'),
    ('right','CONS'),
    ('left', 'PLUS', 'MINUS'),
    ('left','DIVIDE', 'TIMES', 'QDIV','MOD'),
    ('right', 'UMINUS', 'POWER'),
    ('left', 'LBRACKET','RBRACKET','LCURLY','RCURLY'),
    ('left', 'LPAREN','RPAREN','HASH'),
)

# dictionary of varNames
varNameValueIndex = { }

# Tokens
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_LCURLY = r'\{'
t_RCURLY = r'\}'
t_PLUS = r'\+'
t_MINUS = r'-'
t_TIMES = r'\*'
t_DIVIDE = r'/'
t_CONS = r'::'
t_POWER = r'\*\*'
t_LESSERTHAN = '<'
t_LESSEQ = '<='
t_GREATERTHAN = '>'
t_GREATEREQ = '>='
t_DEQUAL = '=='
t_NOTEQUAL = '<>'
t_EQUALS = '='
t_HASH = r'\#'

#are literals needed or can be used as tokens
literals = [',',';']

def t_ID(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = reserved.get(t.value,'NAME')    # Check for reserved words 
    if t.type is 'NAME':
        t.value = VarNode(t.value)
    elif t.type is 'TRUE':
        t.value = BoolNode(True)
    elif t.type is 'FALSE':
        t.value = BoolNode(False)
    return t

def t_NUMBER(t):
    r'\d*(\d\.|\.\d)\d*(e-|e)?\d*|\d+|\d*(e-|e)\d*'
    try:
        t.value = NumberNode(t.value)
    except ValueError:
        raise SemanticError("Integer value too large %d", t.value)
        t.value = 0
    return t

def t_STRING(t):
    r'(\"(\\.|[^"\\])*\")|(\'(\\.|[^\'\\])*\')'
    t.value = StringNode(t.value,1)
    return t

# Ignored characters
t_ignore = " \t"


def t_error(t):
    raise SyntaxError("Syntax error at '%s'" % t.value)

#Parser Functions

# Grammars


#Code for functions HW5
funcNameIndex = {}

class FuncCallNode(Node):
    def __init__(self,name,argValTuple):
        self.name = name #string
        self.argValTuple = argValTuple
    def reduce(self):
        
        global varNameValueIndex
        #Keep a copy of the global var
        bkpVarNameValueIndex = varNameValueIndex
 
        global funcNameIndex
        funcNode = funcNameIndex[self.name]
        #Create the local scope varIndex
        localVarNameValueIndex = {}
        #point glob to loc. This is for child func to access this funcs scope
        for i in range(len(funcNode.argNameTuple.evaluate())):
            x = funcNode.argNameTuple.evaluate()
            localVarNameValueIndex[funcNode.argNameTuple.evaluate()[i].getName()] = self.argValTuple.evaluate()[i].reduceToPrimNode()
        
        varNameValueIndex = localVarNameValueIndex

        funcNode.block.evaluate()
        result = funcNode.output.reduceToPrimNode()

        # result.name += str(randint(1,10000))

        #reset the global varIndex
        varNameValueIndex = bkpVarNameValueIndex
        #Store the curr result in the global dict
        #varNameValueIndex[result.name] = result.reduceToPrimNode()

        return result

class FuncNode(Node):
    def __init__(self,name,argNameTuple,block,output):
        self.name = name
        self.argNameTuple = argNameTuple
        self.block = block
        self.output = output


    def evaluate(self):
        funcNameIndex[self.name] = self
        

def p_superblock_2(t):
    '''
    superblock : superblock block
    '''
    t[0] = BlockNode([t[1]] + [t[2]])


def p_superblock_1(t):
    '''
    superblock : block    
    '''
    t[0] = t[1]


def p_fun(t):
    '''
    block : FUN NAME funcarg EQUALS block NAME ';'
    '''
    #TODO Write a stricter grammar
    t[0] = BlockNode( [FuncNode(t[2].getName(),t[3],t[5],t[6])] )


#End of func code

def p_block_3(t):
    '''
    statement : block
    '''
    t[0] = t[1]


def p_block_2(t):
    '''
    block : LCURLY statement_list RCURLY
    '''
    t[0] = BlockNode(t[2])

def p_block_1(t):
    '''
    block : LCURLY RCURLY
    '''
    t[0] = BlockNode([])


def p_statement_list_2(t):
    '''
    statement_list : statement_list statement
    '''
    t[0] = t[1] + [t[2]]


def p_statement_list_1(t):
    '''
    statement_list : statement 
    '''
    t[0] = [t[1]]

def p_while_statement(t):
    '''
    statement : WHILE LPAREN expression RPAREN block
    '''
    t[0] = WhileNode(t[3],t[5])

def p_if_statement(t):
    '''
    statement : IF LPAREN expression RPAREN block
    '''
    t[0] = IfElseNode(t[3],t[5],None)

def p_ifelse_statement(t):
    '''
    statement : IF LPAREN expression RPAREN block ELSE block
    '''
    t[0] = IfElseNode(t[3],t[5],t[7])

def p_print_statement(t) :
    '''
    statement : PRINT LPAREN expression RPAREN ';'
    '''
    t[0] = PrintNode(t[3])

def p_statment_assign_3(t):
    '''
    statement : NAME LBRACKET expression RBRACKET LBRACKET expression RBRACKET EQUALS expression ';'
    '''
    t[1].pushQueryIndex(t[3])
    t[1].pushQueryIndex(t[6])
    t[0] = varInitOpNode(t[1],t[9])


def p_statement_assign_2(t):
    '''
    statement : NAME LBRACKET expression RBRACKET EQUALS expression ';'
    '''
    t[1].pushQueryIndex(t[3])
    t[0] = varInitOpNode(t[1],t[6])

def p_statement_assign_1(t):
    '''
    statement : NAME EQUALS expression ';'
    '''

    t[0] = varInitOpNode(t[1],t[3])

def p_fun_call_2(t):
    '''
    expression : expression funcarg
    '''
    t[0] = FuncCallNode(t[1].getName(),t[2])

def p_fun_call_1(t):
    '''
    statement : expression funcarg ';'
    '''
    t[0] = FuncCallNode(t[1].getName(),t[2])


def p_funcarg_3(t):
    '''
    funcarg : LPAREN arg RPAREN
    '''
    t[0] = t[2]


def p_funcarg_2(t):
    '''
        arg : arg ',' expression
    '''
    t[0] = TupleNode(t[1].evaluate() + (t[3],) )

def p_funcarg_1(t):
    '''
        arg : expression
    '''
    t[0] = TupleNode((t[1],))





def p_tuple(t):
    '''
    expression : LPAREN in_tuple RPAREN
    '''
    #print (p_tuple.__doc__.strip())
    t[0] = t[2]


def p_empty_tuple(t):
    '''
    expression : LPAREN RPAREN
    '''
    #print(p_empty_tuple.__doc__.strip())
    t[0] = TupleNode(())

def p_in_tuple(t):
    '''
    in_tuple : expression ',' expression
    '''
    #print (p_in_tuple.__doc__.strip())
    t[0] = TupleNode( (t[1],t[3]) )

def p_in_tuple2(t):
    '''
    in_tuple : in_tuple ',' expression
    '''
    #print (p_in_tuple2.__doc__.strip())
    t[0] = appendTupleOp(t[1],t[3])

def p_tuple_item(t):
    '''
    expression : HASH expression expression
    '''
    #print (p_tuple_item.__doc__.strip())
    #t[0] = (t[3].evaluate())[t[2].evaluate()-1]
    t[0] = hashOpNode(t[2],t[3])


def p_in_list(t):
    '''
    in_list : expression
    '''
    #print('in_list : expression')
    t[0] = appendListOpNode(t[1],ListNode([]))


def p_in_list2(t):
    '''
    in_list : in_list ',' expression
    '''
    #print(p_in_list2.__doc__.strip())
    t[0] = appendListOpNode(t[3],t[1])


def p_list(t):
    '''
    expression : LBRACKET in_list RBRACKET
    '''
    #print("expression : LBRACKET in_list RBRACKET")
    t[0] = t[2]

def p_empty_list(t):
    '''
    expression : LBRACKET RBRACKET
    '''
    #print(p_empty_list.__doc__.strip())
    t[0] = ListNode([])


def p_list_item(t):
    '''
    expression : expression LBRACKET expression RBRACKET
    '''
    #print(p_list_item.__doc__.strip())
    t[0] = ListItemOpNode(t[1],t[3])

def p_list_inop(t):
    '''
    expression : expression IN expression
    '''
    #print(p_list_inop.__doc__.strip())
    t[0] = InListOpNode(t[1],t[3])

def p_cons_list(t):
    '''
    expression : expression CONS expression
    '''
    #print(p_cons_list.__doc__.strip())
    t[0] = ConsOpNode(t[1],t[3])

def p_expression_binop(t):
    '''
    expression : expression PLUS expression
                  | expression MINUS expression
                  | expression TIMES expression
                  | expression DIVIDE expression
                  | expression POWER expression
                  | expression QDIV expression
                  | expression MOD expression
                  | expression LESSERTHAN expression
                  | expression LESSEQ expression
                  | expression GREATERTHAN expression
                  | expression GREATEREQ expression
                  | expression AND expression
                  | expression OR expression
                  | expression DEQUAL expression
                  | expression NOTEQUAL expression
    '''
    t[0] = BinopNode(t[2], t[1], t[3])

def p_expression_not(t):
    ''' 
    expression : NOT expression
    '''
    #print(p_expression_not.__doc__.strip())
    t[0] = NotOpNode(t[2])


def p_expression_factor(t):
    '''
    expression : factor
    '''
    #print('expression : factor')
    t[0] = t[1]

def p_factor_number(t):
    '''factor : NUMBER
              | STRING
              | TRUE
              | FALSE
              | NAME
    '''
    #print(p_factor_number.__doc__.strip())
    t[0] = t[1]


def p_expression_uminus(t):
    'expression : MINUS expression %prec UMINUS' #minus get the precedence of uminus
    #print(p_expression_uminus.__doc__.strip())
    t[0] = BinopNode('*', NumberNode(-1) , t[2])

def p_expression_group(t):
    'factor : LPAREN expression RPAREN'
    #print(p_expression_group.__doc__.strip())
    t[0] = t[2]


#End of statements, variableNames Grammar
def p_error(t):
    raise SyntaxError("Did not match any of the Grammars. Syntax error at '%s'" % t.value)

# Build the lexer
import ply.lex as lex

lex.lex(debug=0)

import ply.yacc as yacc

yacc.yacc(debug=0)

testcases_file = sys.argv[1]
#testcases_file = "testcase.txt"

# Open file
with open (testcases_file, "r") as fileHandler:

    code = fileHandler.read().replace('\n', '')

    try:
        lex.input(code)
        while True:
            token = lex.token()
            if not token: break
            # print(token)
        
        try:
            ast = yacc.parse(code)
        except:
            print("SYNTAX ERROR")
            exit(1)
        
        try:
            ast.evaluate()
        except:
            print("SEMANTIC ERROR")
    except Exception as ex:
        if( type(ex).__name__ == 'SyntaxError'):
            print("SYNTAX ERROR")
        else:
            print("SEMANTIC ERROR")
