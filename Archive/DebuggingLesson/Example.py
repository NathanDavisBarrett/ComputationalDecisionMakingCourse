import pyomo.environ as pyo

model = pyo.ConcreteModel()

model.x = pyo.Var()

model.obj = pyo.Objective(expr=model.x)

solver = pyo.SolverFactory("scip")
solver.solve(model,tee=True)




from PyomoTools.IO import ModelToExcel, LoadModelSolutionFromExcel
from PyomoTools import InfeasibilityReport

#STEP 1: Create an excel sheet with a spot for each of the variables in your model.
ModelToExcel(model,"myKnownSolution.xlsx")

#STEP 2: Insert your known solution into the excel sheet
#   Do this in excel

#STEP 3: Load your solution back into the Pyomo model
LoadModelSolutionFromExcel(model,"myKnownSolution.xlsx")

#STEP 4: Identify and print out any violated constraints
report = InfeasibilityReport(model)
report.WriteFile("infeasibilityReport.txt")


from PyomoTools import FindLeastInfeasibleSolution, InfeasibilityReport
#STEP 1: Find the least infeasible solution.
solver = pyo.SolverFactory("scip")
FindLeastInfeasibleSolution(model,solver)

#STEP 2: Identify and print out any violated constraints
report = InfeasibilityReport(model)
report.WriteFile("infeasibilityReport.txt")
