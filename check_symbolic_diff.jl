using Symbolics

# シンボリック変数の定義
@variables u1 u2 u3 u4 w1 w2 w3 w4 mu C

u = [u1, u2, u3, u4]
w = [w1, w2, w3, w4]

# ks_auto_hamilton.jl の K_hamiltonian をシンボリックに定義
T  = (1//8)*(w1^2 + w2^2 + w3^2 + w4^2)
V1 = (1//2)*(-1 + mu - u1^2 + u2^2 + u3^2 - u4^2) * (u2*w1 + u1*w2 - u4*w3 - u3*w4)
V2 = (u1*u2 - u3*u4) * (u1*w1 - u2*w2 - u3*w3 + u4*w4)
V3 = -(1 - mu)*(u1^2 + u2^2 + u3^2 + u4^2) /
     sqrt((u1^2 - u2^2 - u3^2 + u4^2 + 1)^2 + (2u1*u3 + 2u2*u4)^2 + (2u1*u2 - 2u3*u4)^2)
E  = -mu + (1//2) * C * (u1^2 + u2^2 + u3^2 + u4^2)
K  = T + V1 + V2 + V3 + E

println("=" ^ 60)
println("∂K/∂w の計算 (= dzeta[1:4] に対応)")
println("=" ^ 60)

# ∂K/∂w を計算（= dzeta[1:4]）
dK_dw = [Symbolics.derivative(K, wi) for wi in w]

# ks_hamiltonian.jl の解析式
ana_dzeta1 = (1//4)*w1 + (1//2)*(mu - 1 + u1^2 + u2^2 + u3^2 - u4^2)*u2 - u1*u3*u4
ana_dzeta2 = (1//4)*w2 + (1//2)*(mu - 1 - u1^2 - u2^2 + u3^2 - u4^2)*u1 + u2*u3*u4
ana_dzeta3 = (1//4)*w3 + (1//2)*(1 - mu + u1^2 - u2^2 + u3^2 + u4^2)*u4 - u1*u2*u3
ana_dzeta4 = (1//4)*w4 - (1//2)*(mu - 1 - u1^2 + u2^2 + u3^2 + u4^2)*u3 + u1*u2*u4
ana_dw = [ana_dzeta1, ana_dzeta2, ana_dzeta3, ana_dzeta4]

println()
for i in 1:4
    diff = simplify(expand(dK_dw[i] - ana_dw[i]))
    println("dzeta[$i]:")
    println("  シンボリック: $(dK_dw[i])")
    println("  解析式:       $(ana_dw[i])")
    println("  差分 (0なら一致): $diff")
    println()
end

println("=" ^ 60)
println("∂K/∂u の計算 (= -dzeta[5:8] に対応)")
println("=" ^ 60)

# ∂K/∂u を計算（= -dzeta[5:8]）
dK_du = [Symbolics.derivative(K, ui) for ui in u]

# ks_hamiltonian.jl の解析式（dzeta[5:8] は -∂K/∂u なので、∂K/∂u = -dzeta[5:8]）
rE = sqrt((u1^2 - u2^2 - u3^2 + u4^2 + 1)^2 + 4*(u1*u3 + u2*u4)^2 + (2u1*u2 - 2u3*u4)^2)
r  = u1^2 + u2^2 + u3^2 + u4^2

ana_dzeta5 =  -C*u1
              - (1//2)*w2*(mu - 1 - 3*u1^2 - u2^2 + u3^2 - u4^2)
              - 2*(mu - 1)*u1/rE
              + 2*(mu - 1)*u1 * r * (r + 1) / rE^3
              - w1 * (u1*u2 - u3*u4)
              - w3 * (u1*u4 - u2*u3)
              - w4 * (u1*u3 + u2*u4)

ana_dzeta6 =  -C*u2
              - (1//2)*w1*(mu - 1 + u1^2 + 3*u2^2 + u3^2 - u4^2)
              - 2*(mu - 1)*u2/rE
              + 2*(mu - 1)*u2 * r * (r - 1) / rE^3
              + w2 * (u1*u2 - u3*u4)
              + w3 * (u1*u3 + u2*u4)
              - w4 * (u1*u4 - u2*u3)

ana_dzeta7 =  -C*u3
              + (1//2)*w4*(mu - 1 - u1^2 + u2^2 + 3*u3^2 + u4^2)
              - 2*(mu - 1)*u3/rE
              + 2*(mu - 1)*u3 * r * (r - 1) / rE^3
              + w1 * (u1*u4 - u2*u3)
              - w2 * (u1*u3 + u2*u4)
              + w3 * (u1*u2 - u3*u4)

ana_dzeta8 =  -C*u4
              + (1//2)*w3*(mu - 1 - u1^2 + u2^2 - u3^2 - 3*u4^2)
              - 2*(mu - 1)*u4/rE
              + 2*(mu - 1)*u4 * r * (r + 1) / rE^3
              + w1 * (u1*u3 + u2*u4)
              + w2 * (u1*u4 - u2*u3)
              - w4 * (u1*u2 - u3*u4)

ana_du = [-ana_dzeta5, -ana_dzeta6, -ana_dzeta7, -ana_dzeta8]

println()
for i in 1:4
    diff = simplify(expand(dK_du[i] - ana_du[i]))
    println("u[$i] (dK/du[$i] vs -dzeta[$(4+i)]):")
    println("  シンボリック: $(dK_du[i])")
    println("  解析式(-dzeta):  $(ana_du[i])")
    println("  差分 (0なら一致): $diff")
    println()
end

println("=" ^ 60)
println("差分の数値評価（ランダム点でのチェック）")
println("=" ^ 60)
println()

# 数値的に差分を評価して確認（シンボリックの simplify が複雑な場合のため）
using Random
Random.seed!(42)

# テスト点（Moon付近を避けた点）
test_u = [0.3, 0.1, 0.2, 0.0]
test_w = [0.5, -0.3, 0.1, 0.2]
test_mu = 0.01215
test_C  = 3.15

sub_dict = Dict(
    u1 => test_u[1], u2 => test_u[2], u3 => test_u[3], u4 => test_u[4],
    w1 => test_w[1], w2 => test_w[2], w3 => test_w[3], w4 => test_w[4],
    mu => test_mu, C => test_C
)

println("テスト点: u=$(test_u), w=$(test_w), mu=$(test_mu), C=$(test_C)")
println()

println("dzeta[1:4] の比較:")
for i in 1:4
    val_sym = substitute(dK_dw[i], sub_dict)
    val_ana = substitute(ana_dw[i], sub_dict)
    println("  dzeta[$i]: シンボリック=$(Float64(val_sym)), 解析式=$(Float64(val_ana)), 差=$(Float64(val_sym - val_ana))")
end

println()
println("dzeta[5:8] の比較:")
for i in 1:4
    val_sym  = substitute(-dK_du[i], sub_dict)  # dzeta[5:8] = -dK/du
    val_ana  = substitute([ana_dzeta5, ana_dzeta6, ana_dzeta7, ana_dzeta8][i], sub_dict)
    println("  dzeta[$(4+i)]: シンボリック=$(Float64(val_sym)), 解析式=$(Float64(val_ana)), 差=$(Float64(val_sym - val_ana))")
end
