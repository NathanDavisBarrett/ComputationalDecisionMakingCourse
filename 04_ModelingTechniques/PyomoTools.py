import pyomo.environ as pyo
from pyomo.core.expr.current import identify_variables
import re
from typing import Union

import numpy as np

def LoadIndexedSet(model,setName,setDict):
    """
    A function to define an indexed set (dict mapping keys to pyomo Set objects) within a pyomo model.
    
    Once defined you will be able to access each subset using the following syntax:
        model.setName[index]

    Parameters
    ----------
    model: pyo.ConcreteModel
        The model to which you'd like to add the indexed set
    setName: str
        The name of the set you'd like to add. Node that this needs to friendly to python syntax (e.g. no spaces, periods, dashes, etc.)
    setDict: dict (index -> iterable)
        A dict mapping each key to the contents of each corresponding subset.
    """
    setattr(model,setName,{})
    for key in setDict:
        attrName = "{}_{}".format(setName,key)
        setattr(model,attrName,pyo.Set(initialize=setDict[key]))
        getattr(model,setName)[key] = getattr(model,attrName)

def Load2DIndexedSet(model,setName,setDict):
    """
    A function to define a 2-leveled indexed set (dict mapping keys to another dict mapping keys to pyomo Set objects) within a pyomo model.
    
    Once defined you will be able to access each subset using the following syntax:
        model.setName[index1][index2]

    Parameters
    ----------
    model: pyo.ConcreteModel
        The model to which you'd like to add the indexed set
    setName: str
        The name of the set you'd like to add. Node that this needs to friendly to python syntax (e.g. no spaces, periods, dashes, etc.)
    setDict: dict (non-iterable -> iterable)
        A dict mapping each key to another dict mapping each sub-key the contents of each corresponding subset.
    """
    setattr(model,setName,{})
    for k1 in setDict:
        getattr(model,setName)[k1] = {}
        for k2 in setDict[k1]:
            attrName = "{}_{}_{}".format(setName,k1,k2)
            setattr(model,attrName,pyo.Set(initialize=setDict[k1][k2]))
            getattr(model,setName)[k1][k2] = getattr(model,attrName)

def GenerateExpressionStrings(expr):
    """
    A function to generate a pair of strings that represent the given expression.
    Both strings will be the same length and should be printed one above another.
    The first string is the symbolic expression.
    The second string is the expression with each variable name replaced by it's value.

    Parameters
    ----------
    expr: pyomo expression object
        The expression you'd like to generate a string pair for.
    """
    symStr = str(expr)
    numStr = str(expr)

    vrs = list(identify_variables(expr))
    vrs = sorted(vrs,reverse=True,key=lambda v:len(str(v)))

    for v in vrs:
        varStr = v.getname()
        valStr = str(pyo.value(v))

        varStrLen = len(varStr)
        valStrLen = len(valStr)

        if varStrLen >= valStrLen:
            valStr = valStr + " "*(varStrLen-valStrLen)
            numStr = numStr.replace(varStr,valStr)
        else:
            newVarStr = varStr + " "*(valStrLen-varStrLen)
            numStr = numStr.replace(varStr,valStr)
            symStr = symStr.replace(varStr,newVarStr)

    return symStr,numStr

class InfeasibilityReport:
    """
    A class to hold information pertaining to a set of violated constraints.
    
    Members:
    --------
    exprs: 
        A dict mapping (for each violated constraint) constraint names to either that constraint's expression string object or another dict that maps that constraint's indices to those indices' expression strings
    substitutedExprs: 
        A dict with the same structure as exprs but with the value of each variable substituted into the expression string.
    """
    def __init__(self, model:pyo.ConcreteModel,aTol=1e-3):
        self.exprs = {}
        self.substitutedExprs = {}

        self.numInfeas = 0

        for c in model.component_objects(pyo.Constraint, active=True):
            constr = getattr(model,str(c))
            if "Indexed" in str(type(constr)):
                #This constraint is indexed
                for index in constr:
                    if not self.TestFeasibility(constr[index],aTol=aTol):
                        self.AddInfeasibility(name=c,index=index,constr=constr[index])
            else:
                if not self.TestFeasibility(constr,aTol=aTol):
                    self.AddInfeasibility(name=c,constr=constr)

    def TestFeasibility(self,constr,aTol=1e-5):
        lower = constr.lower
        upper = constr.upper
        value = pyo.value(constr)
        
        if lower is not None:
            if value < lower - aTol:
                return False
        if upper is not None:
            if value > upper + aTol:
                return False
        return True

    def AddInfeasibility(self,name:str,constr:pyo.Constraint,index:object=None):
        """
        A function to add a violated constraint to this report

        Parameters:
        -----------
        name: str
            The name of the violated constraint
        constr: pyo.Constraint
            The constraint object
        index: object (optional, Default=None)
            If the constraint is indexed, pass the appropriate index here.
        """
        self.numInfeas += 1
        if index is None:
            self.exprs[name], self.substitutedExprs[name] = GenerateExpressionStrings(constr.expr)
        else:
            if name not in self.exprs:
                self.exprs[name] = {}
                self.substitutedExprs[name] = {}
            self.exprs[name][index], self.substitutedExprs[name][index] = GenerateExpressionStrings(constr.expr)

    def Iterator(self):
        """
        A python generator object (iterator) that iterates over each infeasibility.

        Iterates are strings of the following format: ConstraintName[Index (if appropriate)]: Expr \n SubstitutedExpression
        """
        for c in self.exprs:
            if isinstance(self.exprs[c],dict):
                for i in self.exprs[c]:
                    varName = "{}[{}]:".format(c,i)

                    spaces = " "*len(varName)
                    shortenedStr = re.sub(' +', ' ', self.substitutedExprs[c][i])
                    dividers = ["==","<=",">="]
                    divider = None
                    for d in dividers:
                        if d in shortenedStr:
                            divider = d
                            break
                    
                    if divider is None:
                        raise Exception(f"The following expression is not well posed as a constraint!\n{shortenedStr}")
                    
                    divIndex = shortenedStr.index(divider)
                    lhs = shortenedStr[:divIndex].lstrip()
                    rhs = shortenedStr[divIndex+2:].lstrip()
                    lhsVal = eval(lhs)
                    rhsVal = eval(rhs)

                    evalStr = f"{lhsVal} {divider} {rhsVal}"

                    yield "{} {}\n{} {}\n{} {}\n{} {}".format(varName,self.exprs[c][i],spaces,self.substitutedExprs[c][i],spaces,shortenedStr,spaces,evalStr)
            else:
                spaces = " "*len(c)
                shortenedStr = re.sub(' +', ' ', self.substitutedExprs[c])
                dividers = ["==","<=",">="]
                divider = None
                for d in dividers:
                    if d in shortenedStr:
                        divider = d
                        break
                
                if divider is None:
                    raise Exception(f"The following expression is not well posed as a constraint!\n{shortenedStr}")
                
                divIndex = shortenedStr.index(divider)
                lhs = shortenedStr[:divIndex].lstrip()
                rhs = shortenedStr[divIndex+2:].lstrip()
                lhsVal = eval(lhs)
                rhsVal = eval(rhs)

                evalStr = f"{lhsVal} {divider} {rhsVal}"

                yield "{}: {}\n{}  {}\n{}  {}\n{}  {}".format(c,self.exprs[c],spaces,self.substitutedExprs[c],spaces,shortenedStr,spaces,evalStr)

    def __len__(self):
        return self.numInfeas
    
    def WriteFile(self,fileName:str):
        """
        A function to write the output to a file.
        """
        with open(fileName,"w") as f:
            for infeas in self.Iterator():
                f.write(infeas)
                f.write("\n\n")

def DoubleSidedBigM(
        model:pyo.ConcreteModel,
        A:Union[pyo.Var, pyo.Expression],
        B:Union[pyo.Var, pyo.Expression],
        Bmin:Union[float, dict],
        Bmax:Union[float, dict],
        C:Union[pyo.Var, pyo.Expression, float, dict]=0.0,
        X:Union[pyo.Var, pyo.Expression]=None,
        itrSet:pyo.Set=None,
        includeUpperBounds:bool=True,
        includeLowerBounds:bool=True,
        relationshipBaseName:str=None):
    """
    A function to model the following relationship in MILP form:

        A = X * B + C

    where 
    * A is a Real number
    * B is a Real number
    * C is a Real number, binary, or parameter
    * X is a binary.

    Parameters
    ----------
    model: pyo.ConcreteModel
        The Pyomo model you'd like to instantiate this relationship within
    A: pyo.Var | pyo.Expression
        The Pyomo variable or expression representing "A" in this relationship
    B: pyo.Var | pyo.Expression
        The Pyomo variable or expression representing "B" in this relationship
    Bmin: float | dict
        A float (if no itrSet is provided) or a dict mapping the elements of itrSet to floats (if itrSet is provided) indicating the minimum possible value of "B"
    Bmax: float | dict
        A float (if no itrSet is provided) or a dict mapping the elements of itrSet to floats (if itrSet is provided) indicating the maximum possible value of "B"
    C: pyo.Var | pyo.Expression | float | dict (optional, Default=0.0)
        The value of "C" in this relationship.
    X: pyo.Var | pyo.Expression (optional, Default = None)
        The Pyomo variable or expression representing "X" in this relationship. Note that if "X" is an expression it must evaluate to a binary value in the true feasible space. If None is provided, a unique Binary variable will be generated
    itrSet: pyo.Set (optional, Default=None)
        The set over which to instantiate this relationship. Note that, if provided, A, B, Bmin, Bmax, C, and X must all be defined over this set. If None is provided, this relationship will be instantiated only for the non-indexed instance.
    includeUpperBounds: bool (optional, Default=True)
        An indication of whether or not you'd like to instantiate the upper bounds of this relationship. Only mark this as False if you're certain that "A" will never be maximized.
    includeLowerBounds: bool (optional, Default=True)
        An indication of whether or not you'd like to instantiate the lower bounds of this relationship. Only mark this as False if you're certain that "A" will never be minimized.
    relationshipBaseName: str (optional, Default=None)
        The base name of the generated constraints and variables for this relationship. If None is provided, one will be generated.
        
    Returns
    -------
    tuple:
        lowerBound0: pyo.Constraint | None
            The pyomo constraint representing the lower bound of this relationship if X = 0 (if includeLowerBounds is True)
        lowerBound1: pyo.Constraint | None
            The pyomo constraint representing the lower bound of this relationship if X = 1 (if includeLowerBounds is True)
        upperBound0: pyo.Constraint | None
            The pyomo constraint representing the upper bound of this relationship if X = 0 (if includeUpperBounds is True)
        upperBound1: pyo.Constraint | None
            The pyomo constraint representing the upper bound of this relationship if X = 1 (if includeUpperBounds is True)
        X: pyo.Var | pyo.Expression
            The Pyomo variable expression representing "X" in this relationship.
    """
    if relationshipBaseName is None:
        Aname = str(A)
        Bname = str(B)
        if isinstance(C,float) or isinstance(C,dict):
            Cname = None,
            Caddage = ""
        else:
            Cname = str(C)
            Caddage = f"_{Cname}"

        relationshipBaseName = f"{Aname}_{Bname}{Caddage}_DoubleSidedBigM"

    if isinstance(C,float) and itrSet is not None:
        C = {idx: C for idx in itrSet}

    lowerBound0Name = f"{relationshipBaseName}_lowerBound0"
    lowerBound1Name = f"{relationshipBaseName}_lowerBound1"
    upperBound0Name = f"{relationshipBaseName}_upperBound0"
    upperBound1Name = f"{relationshipBaseName}_upperBound1"

    if X is None:
        Xname = f"{relationshipBaseName}_X"
        if itrSet is None:
            setattr(model,Xname,pyo.Var(domain=pyo.Binary))
            X = getattr(model,Xname)
        else:
            setattr(model,Xname,pyo.Var(itrSet,domain=pyo.Binary))
            X = getattr(model,Xname)

    if itrSet is None:
        if includeLowerBounds:
            setattr(model,lowerBound0Name,pyo.Constraint(expr=A >= Bmin * X + C))
            lowerBound0 = getattr(model,lowerBound0Name)

            setattr(model,lowerBound1Name,pyo.Constraint(expr=A >= B + Bmax*(X-1) + C))
            lowerBound1 = getattr(model,lowerBound1Name)
        else:
            lowerBound0 = None
            lowerBound1 = None
        
        if includeUpperBounds:
            setattr(model,upperBound0Name,pyo.Constraint(expr=A <= Bmax * X + C))
            upperBound0 = getattr(model,upperBound0Name)

            setattr(model,upperBound1Name,pyo.Constraint(expr=A <= B + Bmin*(X-1) + C))
            upperBound1 = getattr(model,upperBound1Name)
        else:
            upperBound0 = None
            upperBound1 = None
    else:
        if includeLowerBounds:
            def lowerBound0Func(model,*idx):
                if len(idx) == 1:
                    idx = idx[0]
                return A[idx] >= Bmin[idx] * X[idx] + C[idx]
            setattr(model,lowerBound0Name,pyo.Constraint(itrSet,rule=lowerBound0Func))
            lowerBound0 = getattr(model,lowerBound0Name)

            def lowerBound1Func(model,*idx):
                if len(idx) == 1:
                    idx = idx[0]
                return A[idx] >= B[idx] + Bmax[idx]*(X[idx]-1) + C[idx]
            setattr(model,lowerBound1Name,pyo.Constraint(itrSet,rule=lowerBound1Func))
            lowerBound1 = getattr(model,lowerBound1Name)
        else:
            lowerBound0 = None
            lowerBound1 = None
        
        if includeUpperBounds:
            def upperBound0Func(model,*idx):
                if len(idx) == 1:
                    idx = idx[0]
                return A[idx] <= Bmax[idx] * X[idx] + C[idx]
            setattr(model,upperBound0Name,pyo.Constraint(itrSet,rule=upperBound0Func))
            upperBound0 = getattr(model,upperBound0Name)

            def upperBound1Func(model,*idx):
                if len(idx) == 1:
                    idx = idx[0]
                return A[idx] <= B[idx] + Bmin[idx]*(X[idx]-1) + C[idx]
            setattr(model,upperBound1Name,pyo.Constraint(itrSet,rule=upperBound1Func))
            upperBound1 = getattr(model,upperBound1Name)
        else:
            upperBound0 = None
            upperBound1 = None

    return (lowerBound0,lowerBound1,upperBound0,upperBound1,X)

def MinOperator(
        model:pyo.ConcreteModel,
        A:Union[pyo.Var, pyo.Expression],
        B:Union[pyo.Var, pyo.Expression],
        C:Union[pyo.Var, pyo.Expression],
        maxDiff:Union[float,dict]=None,
        Y:pyo.Var=None,
        itrSet:pyo.Set=None,
        allowMinimizationPotential:bool=True,
        relationshipBaseName:str=None):
    """
    A function to model the following relationship in MILP or LP form:

        A = min(B,C)

    Parameters
    ----------
    model: pyo.ConcreteModel
        The Pyomo model you'd like to instantiate this relationship within
    A: pyo.Var | pyo.Expression
        The Pyomo variable or expression representing "A" in this relationship
    B: pyo.Var | pyo.Expression
        The Pyomo variable or expression representing "B" in this relationship
    C: pyo.Var | pyo.Expression
        The Pyomo variable or expression representing "B" in this relationship
    maxDiff: float | dict (optional, Default=None)
        The absolute value of the maximum difference between A and B. If no itrSet is provided this must be the float containing the maximum value. If an itrSet is provided, this must be a dict mapping elements of the itrSet to this associated maximum value. Additionally, if allowMinimizationPotential is False, maxDiff can be left as None.
    Y: pyo.Var (optional, Default=None)
        The Pyomo binary variable potentially needed for representing in this relationship. If None is provided and one is needed, a unique Binary variable will be generated
    itrSet: pyo.Set (optional, Default=None)
        The set over which to instantiate this relationship. Note that, if provided, A, B, C,maxDiff and Y must all be defined over this set. If None is provided, this relationship will be instantiated only for the non-indexed instance.
    allowMinimizationPotential: bool (optional, Default=True)
        An indication of whether or not to configure this relationship in such a way to allow "A" to be minimized. If "A" will strictly be maximized, this relationship can simply be modeled as a convex set of two inequality constraints. But if "A" can or will be minimized, this relationship must be modeled using a Binary.
    relationshipBaseName: str (optional, Default=None)
        The base name for the variables and constraints generated by this relationship. If None is provided, one will be generated.

    Returns
    -------
    if allowMinimizationPotential is False:
        tuple of pyo.Constraint:
            bound0: pyo.Constraint
                The bound of A with respect to B
            bound1: pyo.Constraint
                The bound of A with respect to C
    else:
        tuple of pyo.Constraint:
            The constraints necessary to model this relationship
        pyo.Var:
            The "Y" variable used in this relationship.
    """
    if relationshipBaseName is None:
        Aname = str(A)
        Bname = str(B)
        Cname = str(C)
        relationshipBaseName = f"{Aname}_{Bname}_{Cname}_MinOperator"

    if not allowMinimizationPotential:
        bound0Name = f"{relationshipBaseName}_bound0"
        bound1Name = f"{relationshipBaseName}_bound1"
        if itrSet is None:
            setattr(model,bound0Name,pyo.Constraint(expr=A <= B))
            bound0 = getattr(model,bound0Name)

            setattr(model,bound1Name,pyo.Constraint(expr=A <= C))
            bound1 = getattr(model,bound1Name)
        else:
            def bound0Func(model,*idx):
                if len(idx) == 1:
                    idx = idx[0]
                return A[idx] <= B[idx]
            setattr(model,bound0Name,pyo.Constraint(itrSet,rule=bound0Func))
            bound0 = getattr(model,bound0Name)

            def bound1Func(model,*idx):
                if len(idx) == 1:
                    idx = idx[0]
                return A[idx] <= C[idx]
            setattr(model,bound1Name,pyo.Constraint(itrSet,rule=bound1Func))
            bound1 = getattr(model,bound1Name)
        
        return (bound0,bound1)
    else:
        if Y is None:
            Yname = f"{relationshipBaseName}_Y"
            if itrSet is None:
                setattr(model,Yname,pyo.Var(domain=pyo.Binary))
                Y = getattr(model,Yname)
            else:
                setattr(model,Yname,pyo.Var(itrSet,domain=pyo.Binary))
                Y = getattr(model,Yname)

        c0Name = f"{relationshipBaseName}_constraint0"
        c1Name = f"{relationshipBaseName}_constraint1"
        c2Name = f"{relationshipBaseName}_constraint2"
        c3Name = f"{relationshipBaseName}_constraint3"
        c4Name = f"{relationshipBaseName}_constraint4"
        c5Name = f"{relationshipBaseName}_constraint5"

        if itrSet is None:
            M = np.abs(maxDiff)

            setattr(model,c0Name,pyo.Constraint(expr=C-B <= M * Y))
            c0 = getattr(model,c0Name)

            setattr(model,c1Name,pyo.Constraint(expr=B-C <= M * (1-Y)))
            c1 = getattr(model,c1Name)

            setattr(model,c2Name,pyo.Constraint(expr=A <= B))
            c2 = getattr(model,c2Name)

            setattr(model,c3Name,pyo.Constraint(expr=A <= C))
            c3 = getattr(model,c3Name)

            setattr(model,c4Name,pyo.Constraint(expr=A >= B - M * (1-Y)))
            c4 = getattr(model,c4Name)

            setattr(model,c5Name,pyo.Constraint(expr=A >= C - M * Y))
            c5 = getattr(model,c5Name)
        else:
            M = {idx: np.abs(maxDiff[idx]) for idx in itrSet}

            def c0Func(model,*idx):
                if len(idx) == 1:
                    idx = idx[0]
                return C[idx] - B[idx] <= M[idx] * Y[idx]
            setattr(model,c0Name,pyo.Constraint(itrSet,rule=c0Func))
            c0 = getattr(model,c0Name)

            def c1Func(model,*idx):
                if len(idx) == 1:
                    idx = idx[0]
                return B[idx] - C[idx] <= M[idx] * (1-Y[idx])
            setattr(model,c1Name,pyo.Constraint(itrSet,rule=c1Func))
            c1 = getattr(model,c1Name)

            def c2Func(model,*idx):
                if len(idx) == 1:
                    idx = idx[0]
                return A[idx] <= B[idx]
            setattr(model,c2Name,pyo.Constraint(itrSet,rule=c2Func))
            c2 = getattr(model,c2Name)

            def c3Func(model,*idx):
                if len(idx) == 1:
                    idx = idx[0]
                return A[idx] <= C[idx]
            setattr(model,c3Name,pyo.Constraint(itrSet,rule=c3Func))
            c3 = getattr(model,c3Name)

            def c4Func(model,*idx):
                if len(idx) == 1:
                    idx = idx[0]
                return A[idx] >= B[idx] - M[idx] * (1-Y[idx])
            setattr(model,c4Name,pyo.Constraint(itrSet,rule=c4Func))
            c4 = getattr(model,c4Name)

            def c5Func(model,*idx):
                if len(idx) == 1:
                    idx = idx[0]
                return A[idx] >= C[idx] - M[idx] * Y[idx]
            setattr(model,c5Name,pyo.Constraint(itrSet,rule=c5Func))
            c5 = getattr(model,c5Name)
        
        return (c0,c1,c2,c3,c4,c5)

def MaxOperator(
        model:pyo.ConcreteModel,
        A:Union[pyo.Var, pyo.Expression],
        B:Union[pyo.Var, pyo.Expression],
        C:Union[pyo.Var, pyo.Expression],
        maxDiff:Union[float,dict]=None,
        Y:pyo.Var=None,
        itrSet:pyo.Set=None,
        allowMaximizationPotential:bool=True,
        relationshipBaseName:str=None):
    """
    A function to model the following relationship in MILP or LP form:

        A = max(B,C)

    Parameters
    ----------
    model: pyo.ConcreteModel
        The Pyomo model you'd like to instantiate this relationship within
    A: pyo.Var | pyo.Expression
        The Pyomo variable or expression representing "A" in this relationship
    B: pyo.Var | pyo.Expression
        The Pyomo variable or expression representing "B" in this relationship
    C: pyo.Var | pyo.Expression
        The Pyomo variable or expression representing "B" in this relationship
    maxDiff: float | dict (optional, Default=None)
        The absolute value of the maximum difference between A and B. If no itrSet is provided this must be the float containing the maximum value. If an itrSet is provided, this must be a dict mapping elements of the itrSet to this associated maximum value. Additionally, if allowMinimizationPotential is False, maxDiff can be left as None.
    Y: pyo.Var (optional, Default=None)
        The Pyomo binary variable potentially needed for representing in this relationship. If None is provided and one is needed, a unique Binary variable will be generated
    itrSet: pyo.Set (optional, Default=None)
        The set over which to instantiate this relationship. Note that, if provided, A, B, C,maxDiff and Y must all be defined over this set. If None is provided, this relationship will be instantiated only for the non-indexed instance.
    allowMaximizationPotential: bool (optional, Default=True)
        An indication of whether or not to configure this relationship in such a way to allow "A" to be maximized. If "A" will strictly be minimized, this relationship can simply be modeled as a convex set of two inequality constraints. But if "A" can or will be maximized, this relationship must be modeled using a Binary.
    relationshipBaseName: str (optional, Default=None)
        The base name for the variables and constraints generated by this relationship. If None is provided, one will be generated.

    Returns
    -------
    if allowMaximizationPotential is False:
        tuple of pyo.Constraint:
            bound0: pyo.Constraint
                The bound of A with respect to B
            bound1: pyo.Constraint
                The bound of A with respect to C
    else:
        tuple of pyo.Constraint:
            The constraints necessary to model this relationship
        pyo.Var:
            The "Y" variable used in this relationship.
    """
    if relationshipBaseName is None:
        Aname = str(A)
        Bname = str(B)
        Cname = str(C)
        relationshipBaseName = f"{Aname}_{Bname}_{Cname}_MaxOperator"

    if not allowMaximizationPotential:
        bound0Name = f"{relationshipBaseName}_bound0"
        bound1Name = f"{relationshipBaseName}_bound1"
        if itrSet is None:
            setattr(model,bound0Name,pyo.Constraint(expr=A >= B))
            bound0 = getattr(model,bound0Name)

            setattr(model,bound1Name,pyo.Constraint(expr=A >= C))
            bound1 = getattr(model,bound1Name)
        else:
            def bound0Func(model,*idx):
                if len(idx) == 1:
                    idx = idx[0]
                return A[idx] >= B[idx]
            setattr(model,bound0Name,pyo.Constraint(itrSet,rule=bound0Func))
            bound0 = getattr(model,bound0Name)

            def bound1Func(model,*idx):
                if len(idx) == 1:
                    idx = idx[0]
                return A[idx] >= C[idx]
            setattr(model,bound1Name,pyo.Constraint(itrSet,rule=bound1Func))
            bound1 = getattr(model,bound1Name)
        
        return (bound0,bound1)
    else:
        if Y is None:
            Yname = f"{relationshipBaseName}_Y"
            if itrSet is None:
                setattr(model,Yname,pyo.Var(domain=pyo.Binary))
                Y = getattr(model,Yname)
            else:
                setattr(model,Yname,pyo.Var(itrSet,domain=pyo.Binary))
                Y = getattr(model,Yname)

        c0Name = f"{relationshipBaseName}_constraint0"
        c1Name = f"{relationshipBaseName}_constraint1"
        c2Name = f"{relationshipBaseName}_constraint2"
        c3Name = f"{relationshipBaseName}_constraint3"
        c4Name = f"{relationshipBaseName}_constraint4"
        c5Name = f"{relationshipBaseName}_constraint5"

        if itrSet is None:
            M = np.abs(maxDiff)

            setattr(model,c0Name,pyo.Constraint(expr=B-C <= M * Y))
            c0 = getattr(model,c0Name)

            setattr(model,c1Name,pyo.Constraint(expr=C-B <= M * (1-Y)))
            c1 = getattr(model,c1Name)

            setattr(model,c2Name,pyo.Constraint(expr=A >= B))
            c2 = getattr(model,c2Name)

            setattr(model,c3Name,pyo.Constraint(expr=A >= C))
            c3 = getattr(model,c3Name)

            setattr(model,c4Name,pyo.Constraint(expr=A <= B + M * (1-Y)))
            c4 = getattr(model,c4Name)

            setattr(model,c5Name,pyo.Constraint(expr=A <= C + M * Y))
            c5 = getattr(model,c5Name)
        else:
            M = {idx: np.abs(maxDiff[idx]) for idx in itrSet}

            def c0Func(model,*idx):
                if len(idx) == 1:
                    idx = idx[0]
                return B[idx] - C[idx] <= M[idx] * Y[idx]
            setattr(model,c0Name,pyo.Constraint(itrSet,rule=c0Func))
            c0 = getattr(model,c0Name)

            def c1Func(model,*idx):
                if len(idx) == 1:
                    idx = idx[0]
                return C[idx] - B[idx] <= M[idx] * (1-Y[idx])
            setattr(model,c1Name,pyo.Constraint(itrSet,rule=c1Func))
            c1 = getattr(model,c1Name)

            def c2Func(model,*idx):
                if len(idx) == 1:
                    idx = idx[0]
                return A[idx] >= B[idx]
            setattr(model,c2Name,pyo.Constraint(itrSet,rule=c2Func))
            c2 = getattr(model,c2Name)

            def c3Func(model,*idx):
                if len(idx) == 1:
                    idx = idx[0]
                return A[idx] >= C[idx]
            setattr(model,c3Name,pyo.Constraint(itrSet,rule=c3Func))
            c3 = getattr(model,c3Name)

            def c4Func(model,*idx):
                if len(idx) == 1:
                    idx = idx[0]
                return A[idx] <= B[idx] + M[idx] * (1-Y[idx])
            setattr(model,c4Name,pyo.Constraint(itrSet,rule=c4Func))
            c4 = getattr(model,c4Name)

            def c5Func(model,*idx):
                if len(idx) == 1:
                    idx = idx[0]
                return A[idx] <= C[idx] + M[idx] * Y[idx]
            setattr(model,c5Name,pyo.Constraint(itrSet,rule=c5Func))
            c5 = getattr(model,c5Name)
        
        return (c0,c1,c2,c3,c4,c5)