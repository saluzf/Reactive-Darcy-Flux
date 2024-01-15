const USE_GPU = false
using ImplicitGlobalGrid
import MPI
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using Printf, MAT
max_g(A) = (max_l = maximum(A); MPI.Allreduce(max_l, MPI.MAX, MPI.COMM_WORLD))
@views av1(A) = 0.5 .* (A[1:end-1] .+ A[2:end])

"""
Saves array as Aname.bin
"""
function save_array(Aname, A)
    out = open(fname, "w")
    write(out, A)
    close(out)
end
"""
Compute pressure fluxes for the fluid
"""
@parallel function compute_Dflux!(qDx, qDy, qDz, Pf, T, k_ηf, _dx, _dy, _dz, αρg, _1_θ_dτ_D)
    @inn_x(qDx) = @inn_x(qDx) - (@inn_x(qDx) + k_ηf * (@d_xa(Pf) * _dx)) * _1_θ_dτ_D
    @inn_y(qDy) = @inn_y(qDy) - (@inn_y(qDy) + k_ηf * (@d_ya(Pf) * _dy)) * _1_θ_dτ_D
    @inn_z(qDz) = @inn_z(qDz) - (@inn_z(qDz) + k_ηf * (@d_za(Pf) * _dz - αρg * @av_za(T))) * _1_θ_dτ_D
    return nothing
end
"""
Updates the pressure
"""
@parallel function update_Pf!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ_D)
    @all(Pf) = @all(Pf) - (@d_xa(qDx) * _dx + @d_ya(qDy) * _dy + @d_za(qDz) * _dz) * _β_dτ_D
    return nothing
end
"""compute temperature fluxes"""
@parallel_indices (ix, iy, iz) function compute_Tflux!(qTx, qTy, qTz, dTdt, T, T_old, qDx, qDy, qDz, _dt, λ_ρCp_dx, λ_ρCp_dy, λ_ρCp_dz, _1_θ_dτ_T, _dx, _dy, _dz, _ϕ)
    if (ix <= size(qTx, 1) && iy <= size(qTx, 2) && iz <= size(qTx, 3))
        qTx[ix, iy, iz] = qTx[ix, iy, iz] - (qTx[ix, iy, iz] + λ_ρCp_dx * (T[ix+1, iy+1, iz+1] - T[ix, iy+1, iz+1])) * _1_θ_dτ_T
    end
    if (ix <= size(qTy, 1) && iy <= size(qTy, 2) && iz <= size(qTy, 3))
        qTy[ix, iy, iz] = qTy[ix, iy, iz] - (qTy[ix, iy, iz] + λ_ρCp_dy * (T[ix+1, iy+1, iz+1] - T[ix+1, iy, iz+1])) * _1_θ_dτ_T
    end
    if (ix <= size(qTz, 1) && iy <= size(qTz, 2) && iz <= size(qTz, 3))
        qTz[ix, iy, iz] = qTz[ix, iy, iz] - (qTz[ix, iy, iz] + λ_ρCp_dz * (T[ix+1, iy+1, iz+1] - T[ix+1, iy+1, iz])) * _1_θ_dτ_T
    end

    if (ix <= size(dTdt, 1) && iy <= size(dTdt, 2) && iz <= size(dTdt, 3))
        dTdt[ix, iy, iz] = (T[ix+1, iy+1, iz+1] - T_old[ix+1, iy+1, iz+1]) * _dt +
                           (max(qDx[ix+1, iy+1, iz+1], 0.0) * (T[ix+1, iy+1, iz+1] - T[ix, iy+1, iz+1]) * _dx +
                            min(qDx[ix+2, iy+1, iz+1], 0.0) * (T[ix+2, iy+1, iz+1] - T[ix+1, iy+1, iz+1]) * _dx +
                            max(qDy[ix+1, iy+1, iz+1], 0.0) * (T[ix+1, iy+1, iz+1] - T[ix+1, iy, iz+1]) * _dy +
                            min(qDy[ix+1, iy+2, iz+1], 0.0) * (T[ix+1, iy+2, iz+1] - T[ix+1, iy+1, iz+1]) * _dy +
                            max(qDz[ix+1, iy+1, iz+1], 0.0) * (T[ix+1, iy+1, iz+1] - T[ix+1, iy+1, iz]) * _dz +
                            min(qDz[ix+1, iy+1, iz+2], 0.0) * (T[ix+1, iy+1, iz+2] - T[ix+1, iy+1, iz+1]) * _dz) * _ϕ
    end
    return nothing
end
"""Update the temperature"""
@parallel function update_T!(T, qTx, qTy, qTz, dTdt, _dx, _dy, _dz, _1_dt_β_dτ_T)
    @inn(T) = @inn(T) - (@all(dTdt) + @d_xa(qTx) * _dx + @d_ya(qTy) * _dy + @d_za(qTz) * _dz) * _1_dt_β_dτ_T
    return nothing
end
"""Computes residuals of pde"""
@parallel function compute_r!(r_Pf, r_T, qDx, qDy, qDz, qTx, qTy, qTz, dTdt, _dx, _dy, _dz)
    @all(r_Pf) = @d_xa(qDx) * _dx + @d_ya(qDy) * _dy + @d_za(qDz) * _dz
    @all(r_T)  = @all(dTdt) + @d_xa(qTx) * _dx + @d_ya(qTy) * _dy + @d_za(qTz) * _dz
    return nothing
end
"""Apply boundary conditions in x direction"""
@parallel_indices (iy, iz) function bc_x!(A)
    A[1  , iy, iz] = A[2    , iy, iz]
    A[end, iy, iz] = A[end-1, iy, iz]
    return
end
"""Apply boundary conditions in y direction"""
@parallel_indices (ix, iz) function bc_y!(A)
    A[ix, 1  , iz] = A[ix, 2    , iz]
    A[ix, end, iz] = A[ix, end-1, iz]
    return
end
@views function porous_convection_3D(; nz=63, do_viz=false)
    # physics
    lx, ly, lz = 40.0, 20.0, 20.0
    k_ηf       = 1.0
    αρg        = 1.0
    ΔT         = 200.0
    ϕ          = 0.1
    Ra         = 1000
    λ_ρCp      = 1 / Ra * (αρg * k_ηf * ΔT * lz / ϕ) # Ra = αρg*k_ηf*ΔT*lz/λ_ρCp/ϕ
    # numerics
    nx,ny       = 2 * (nz + 1) - 1, nz
    me, dims    = init_global_grid(nx, ny, nz)  # init global grid and more
    b_width     = (8, 8, 4)                     # for comm / comp overlap
    nt          = 500
    re_D        = 4π
    cfl         = 1.0 / sqrt(3.1)
    maxiter     = 8max(nx_g(), ny_g(), nz_g())
    ϵtol        = 1e-6
    nvis        = 50
    ncheck      = ceil(2max(nx_g(), ny_g(), nz_g()))
    # preprocessing
    dx, dy, dz = lx / nx_g(), ly / ny_g(), lz / nz_g()
    xn, yn, zn = LinRange(-lx / 2, lx / 2, nx + 1), LinRange(-ly / 2, ly / 2, ny + 1), LinRange(-lz, 0, nz + 1)
    xc, yc, zc = av1(xn), av1(yn), av1(zn)
    _dx, _dy, _dz = 1.0 / dx, 1.0 / dy, 1.0 / dz
    _ϕ         = 1.0 / ϕ
    θ_dτ_D     = max(lx, ly, lz) / re_D / cfl / min(dx, dy, dz)
    β_dτ_D     = (re_D * k_ηf) / (cfl * min(dx, dy, dz) * max(lx, ly, lz))
    _1_θ_dτ_D  = 1.0 / (1.0 + θ_dτ_D)
    _β_dτ_D    = 1.0 / β_dτ_D
    λ_ρCp_dx, λ_ρCp_dy, λ_ρCp_dz = λ_ρCp / dx, λ_ρCp / dy, λ_ρCp / dz
    # init
    Pf      = @zeros(nx, ny, nz)
    r_Pf    = @zeros(nx, ny, nz)
    qDx     = @zeros(nx + 1, ny, nz)
    qDy     = @zeros(nx, ny + 1, nz)
    qDz     = @zeros(nx, ny, nz + 1)
    dTdt    = @zeros(nx - 2, ny - 2, nz - 2)
    r_T     = @zeros(nx - 2, ny - 2, nz - 2)
    qTx     = @zeros(nx - 1, ny - 2, nz - 2)
    qTy     = @zeros(nx - 2, ny - 1, nz - 2)
    qTz     = @zeros(nx - 2, ny - 2, nz - 1)
    T       = @zeros(nx, ny, nz)
    T      .= Data.Array([ΔT * exp(-(x_g(ix, dx, T) + dx / 2 - lx / 2)^2
                              -(y_g(iy, dy, T) + dy / 2 - ly / 2)^2
                              -(z_g(iz, dz, T) + dz / 2 - lz / 2)^2) for ix = 1:size(T, 1), iy = 1:size(T, 2), iz = 1:size(T, 3)])
    T[:, :, 1  ] .=  ΔT / 2
    T[:, :, end] .= -ΔT / 2
    update_halo!(T)
    T_old = copy(T)
    # vis
    if do_viz
        ENV["GKSwstype"]="nul"; if isdir("viz3D_out")==false mkdir("viz3D_out") end
        iframe = 0
        nx_v, ny_v, nz_v = (nx - 2) * dims[1], (ny - 2) * dims[2], (nz - 2) * dims[3]
        T_v   = zeros(nx_v, ny_v, nz_v) # global array for visu
        T_inn = zeros(nx - 2, ny - 2, nz - 2) # no halo local array for visu
        xi_g, zi_g = LinRange(-lx / 2 + dx + dx / 2, lx / 2 - dx - dx / 2, nx_v), LinRange(-lz + dz + dz / 2, -dz - dz / 2, nz_v) # inner points only
    end
    # action
    for it = 1:nt
        T_old = copy(T)
        # time step
        dt = if it == 1
            0.1 * min(dx, dy, dz) / (αρg * ΔT * k_ηf)
        else
            min(5.0 * min(dx, dy, dz) / (αρg * ΔT * k_ηf), ϕ * min(dx / max_g(abs.(qDx)), dy / max_g(abs.(qDy)), dz / max_g(abs.(qDz))) / 3.1)
        end
        _dt    = 1.0 / dt
        re_T   = π + sqrt(π^2 + ly^2 / λ_ρCp / dt)
        θ_dτ_T = max(lx, ly) / re_T / cfl / min(dx, dy)
        β_dτ_T = (re_T * λ_ρCp) / (cfl * min(dx, dy, dz) * max(lx, ly, lz))
        _1_θ_dτ_T    = 1.0 / (1.0 + θ_dτ_T)
        _1_dt_β_dτ_T = 1.0 / (1.0 / dt + β_dτ_T)
        # iteration loop
        iter = 1; err_D = 2ϵtol; err_T = 2ϵtol
        while max(err_D, err_T) >= ϵtol && iter <= maxiter
            # hydro
            @hide_communication b_width begin
                @parallel compute_Dflux!(qDx, qDy, qDz, Pf, T, k_ηf, _dx, _dy, _dz, αρg, _1_θ_dτ_D)
                update_halo!(qDx, qDy, qDz)
            end
            @parallel update_Pf!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ_D)
            
            # thermo
            @parallel compute_Tflux!(qTx, qTy, qTz, dTdt, T, T_old, qDx, qDy, qDz, _dt, λ_ρCp_dx, λ_ρCp_dy, λ_ρCp_dz, _1_θ_dτ_T, _dx, _dy, _dz, _ϕ)
            @hide_communication b_width begin
                @parallel update_T!(T, qTx, qTy, qTz, dTdt, _dx, _dy, _dz, _1_dt_β_dτ_T)
                @parallel (1:size(T, 2), 1:size(T, 3)) bc_x!(T)
                @parallel (1:size(T, 1), 1:size(T, 3)) bc_y!(T)
                update_halo!(T)
            end
            if iter % ncheck == 0 
                @parallel compute_r!(r_Pf, r_T, qDx, qDy, qDz, qTx, qTy, qTz, dTdt, _dx, _dy, _dz)
                err_D = max_g(abs.(r_Pf)); err_T = max_g(abs.(r_T))
                @printf("  iter/nx=%.1f, err_D=%1.3e, err_T=%1.3e\n", iter / nx_g(), err_D, err_T)
            end
            iter += 1
        end
        if me == 0
            @printf("it = %d, iter/nx=%.1f, err_D=%1.3e, err_T=%1.3e\n", it, iter / nx, err_D, err_T)
        end
        # visualisation
        if do_viz &&  (it % nvis == 0)
            T_inn .= Array(T)[2:end-1, 2:end-1, 2:end-1]; gather!(T_inn, T_v)
            if me == 0
                file = matopen("$(@__DIR__)/viz3D_out/mpi3D_out_C_$(it).mat", "w"); write(file, "T", Array(T_v)); close(file)
            end
        end
    end
    finalize_global_grid()
    return
end

porous_convection_3D(; nz=21, do_viz=true)
