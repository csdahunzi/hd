# ======================================================================
#         DVConstraint Setup
# ====================================================================== 
DVCon = DVConstraints_Mode()
DVCon.setDVGeo(DVGeo)

# Only ADflow has the getTriangulatedSurface Function
#DVCon.setSurface(CFDSolver.getTriangulatedMeshSurface())
le=0.1
leList = [[le    , 0, 0.0], [le    , 0, 1.0]]
teList = [[1.0-le, 0, 0.0], [1.0-le, 0, 1.0]]
# Thickness constraints
DVCon.addThicknessConstraints2D(le, 1.0-le, 2, 5, lower=0.9)
#DVCon.addThicknessConstraintsTE( 2, 2, lower=0.5)
#DVCon.addThicknessConstraints2Dvector(leList, teList, 2, 50, lower=0.1)

if comm.rank == 0:
    fileName = os.path.join(args.output, 'constraints.dat')
    DVCon.writeTecplot(fileName)
