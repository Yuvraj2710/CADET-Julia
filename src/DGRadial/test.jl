# test.jl

# Include the main module file
include("radial_dg.jl")
using .RadialLRMDG  # Note the dot to access the module

# Run the simulation
dc = simulate_radial_LRM(N=4, Ne=10, rhomin=0.5, rhomax=1.0, D_rad=1e-4, τ=5.0)

# Print a summary of the result
println("Residual (dc) norm: ", norm(dc))
println("Residual vector:")
println(dc)