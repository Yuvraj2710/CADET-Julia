module RadialLRMDG

using LinearAlgebra
using SparseArrays

# ---- 1. Load Element Utilities (from DGElements, adapted) ----
include("DGElements.jl")  # Mass, diff, boundary, etc.

# ---- 2. DG Element Struct for Radial Geometry ----
struct DGElement
    M::Matrix{Float64}           # Mass matrix
    D::Matrix{Float64}           # Differentiation matrix
    B::Matrix{Float64}           # Boundary matrix
    Mrho::Matrix{Float64}          # Radial-weighted mass matrix
    V::Matrix{Float64}           # Velocity-weighted mass matrix
    rho::Vector{Float64}           # Radial coordinate at LGL nodes
    w::Vector{Float64}           # LGL quadrature weights
end

"""
Build a DG element for radial flow over a given interval (rho₁, rho₂)
"""
function create_radial_element(rho₁, rho₂, velocity_fn, N)
    xi, w = LGL(N)                       # Reference coordinates
    rho = rho₁ .+ (rho₂ - rho₁)/2 .* (1 .+ xi)   # Physical map

    l, dl = lagrange_basis(xi)
    M = diagm(w)                        # Mass matrix
    D = compute_Dmatrix(dl)           # Diff matrix
    B = compute_Bmatrix(l)            # Boundary matrix

    V = Diagonal(w .* velocity_fn.(rho))
    Mrho = Diagonal(w .* rho)

    return DGElement(M, D, B, Mrho, V, rho, w)
end

# ---- 3. Numerical Fluxes ----

"""
Convective upwind flux for radial direction
"""
function flux_c(v, c⁻, c⁺)
    return v > 0 ? v * c⁻ : v * c⁺
end

"""
Symmetric Interior Penalty diffusion flux
"""
function flux_g(rho, g⁻, g⁺, c⁻, c⁺, τ)
    return 0.5 * (rho * (g⁻ + g⁺)) - τ * (c⁺ - c⁻)
end

# ---- 4. Element-local residuals for c and g ----

function element_residual!(dc, g, elem::DGElement, c, D_rad, Δrho, τ, c⁺, g⁺)
    Mrho⁻¹ = inv(elem.Mrho)

    # Compute auxiliary gradient g = ...
    g_tmp = elem.D * c - inv(elem.M) * (elem.B * (c - c⁺))
    g .= D_rad * 2/Δrho * g_tmp

    # Compute volume term: D (V c - Mrho g)
    vol = elem.D * (elem.V * c - elem.Mrho * g)

    # Numerical flux h* = f* - g*
    fc_star = flux_c(elem.rho[end], c[end], c⁺[1])   # outflow edge
    gc_star = flux_g(elem.rho[end], g[end], g⁺[1], c[end], c⁺[1], τ)
    hR = fc_star - gc_star

    fc_star = flux_c(elem.rho[1], c[1], c⁺[2])       # inflow edge
    gc_star = flux_g(elem.rho[1], g[1], g⁺[2], c[1], c⁺[2], τ)
    hL = fc_star - gc_star

    # Build B(h*) vector
    Bflux = elem.B * [hL; zeros(length(c)-2); hR]

    dc .= 2/Δrho * (Mrho⁻¹ * (vol - Bflux))
end

# ---- 5. Top-level: Simulate single radial LRM pulse ----

function simulate_radial_LRM(;N=4, Ne=20, rhomin=0.5, rhomax=1.0, D_rad=1e-4, τ=5.0, T=10.0)
    Δrho = (rhomax - rhomin) / Ne
    elems = [create_radial_element(rhomin+i*Δrho, rhomin+(i+1)*Δrho, rho -> 1.0, N) for i in 0:Ne-1]

    dof = (N+1) * Ne
    c = zeros(dof)          # concentration
    g = zeros(dof)          # aux gradient
    dc = similar(c)         # derivative output

    # ---- Dummy loop for residual evaluation ----
    for (i, elem) in enumerate(elems)
        idx = (i-1)*(N+1)+1 : i*(N+1)
        c⁺ = [0.0, 0.0]  # boundary states (from neighbor or BC)
        g⁺ = [0.0, 0.0]

        if i > 1
            c⁺[2] = c[idx[1] - 1]
            g⁺[2] = g[idx[1] - 1]
        end
        if i < Ne
            c⁺[1] = c[idx[end] + 1]
            g⁺[1] = g[idx[end] + 1]
        end

        element_residual!(dc[idx], g[idx], elem, c[idx], D_rad, Δrho, τ, c⁺, g⁺)
    end

    return dc
end

end # module