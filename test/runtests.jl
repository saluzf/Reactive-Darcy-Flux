using FinalProject
using Test


@testset "FinalProject.jl" begin
    exename = joinpath(Sys.BINDIR, Base.julia_exename())
    testdir = pwd()
    # run test scripts:
    run(`$exename -O3 --startup-file=no $(joinpath(testdir, "testIsothermalSalineAquifier.jl"))`)
    run(`$exename -O3 --startup-file=no $(joinpath(testdir, "testSalineAquifier.jl"))`)
    run(`$exename -O3 --startup-file=no $(joinpath(testdir, "testReactiveSalineAquifier.jl"))`)
end
