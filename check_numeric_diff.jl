using ForwardDiff
include("ks_hamiltonian.jl")

# ks_auto_hamilton.jl のハミルトニアン（そのままコピー）
function K_hamiltonian(u, w, mu, C)
    T = (1/8)*(w[1]^2 + w[2]^2 + w[3]^2 + w[4]^2)
    V1 = (1/2)*(-1 + mu - u[1]^2 + u[2]^2 + u[3]^2 - u[4]^2) * (u[2]*w[1] + u[1]*w[2] - u[4]*w[3] - u[3]*w[4])
    V2 = (u[1]*u[2]-u[3]*u[4]) * (u[1]*w[1] - u[2]*w[2] - u[3]*w[3] + u[4]*w[4])
    V3 = -(1 - mu)*(u[1]^2 + u[2]^2 + u[3]^2 + u[4]^2)/sqrt((u[1]^2 - u[2]^2 - u[3]^2 + u[4]^2 + 1)^2 + (2u[1]*u[3] + 2u[2]*u[4])^2 + (2u[1]*u[2] - 2u[3]*u[4])^2)
    E = -mu + (1/2) * C * (u[1]^2 + u[2]^2 + u[3]^2 + u[4]^2)
    return T + V1 + V2 + V3 + E
end

mu = 0.01215058426994
C  = 3.15

println("=" ^ 70)
println("数値比較: 自動微分 (auto) vs 解析式 (ana) [ks_hamiltonian.jl を直接使用]")
println("=" ^ 70)

test_cases = [
    ([0.3, 0.1, 0.2, 0.0],   [0.5, -0.3, 0.1, 0.2]),
    ([0.5, -0.2, 0.1, 0.3],  [-0.4, 0.2, -0.3, 0.1]),
    ([0.1, 0.4, -0.1, 0.2],  [0.3, 0.1, -0.2, 0.4]),
    ([0.2, 0.0, 0.3, -0.1],  [-0.1, 0.5, 0.0, -0.3]),
]

all_match_dw = true
all_match_du = true

for (k, (u_test, w_test)) in enumerate(test_cases)
    zeta_test = vcat(u_test, w_test)

    # 自動微分
    dK_dw = ForwardDiff.gradient(w -> K_hamiltonian(u_test, w, mu, C), w_test)
    dK_du = ForwardDiff.gradient(u -> K_hamiltonian(u, w_test, mu, C), u_test)

    # ks_hamiltonian.jl の解析式（実際のファイルを使用）
    dzeta_ana = zeros(8)
    ks_hamiltonian(dzeta_ana, zeta_test, [mu, C], 0.0)

    println("\nテスト点 $k: u=$(u_test), w=$(w_test)")
    println("  dzeta[1:4] の比較:")
    for i in 1:4
        diff = dK_dw[i] - dzeta_ana[i]
        status = abs(diff) < 1e-10 ? "✓" : "✗ MISMATCH"
        if abs(diff) > 1e-10; global all_match_dw = false; end
        println("    dzeta[$i]: auto=$(round(dK_dw[i], sigdigits=6)),  ana=$(round(dzeta_ana[i], sigdigits=6)),  差=$(round(diff, sigdigits=3))  $status")
    end
    println("  dzeta[5:8] の比較:")
    for i in 1:4
        auto_val = -dK_du[i]
        diff = auto_val - dzeta_ana[4+i]
        status = abs(diff) < 1e-10 ? "✓" : "✗ MISMATCH"
        if abs(diff) > 1e-10; global all_match_du = false; end
        println("    dzeta[$(4+i)]: auto=$(round(auto_val, sigdigits=6)),  ana=$(round(dzeta_ana[4+i], sigdigits=6)),  差=$(round(diff, sigdigits=3))  $status")
    end
end

println()
println("=" ^ 70)
println("まとめ")
println("=" ^ 70)
println("dzeta[1:4]: ", all_match_dw ? "全テスト点で一致 ✓" : "不一致あり ✗")
println("dzeta[5:8]: ", all_match_du ? "全テスト点で一致 ✓" : "不一致あり ✗")
