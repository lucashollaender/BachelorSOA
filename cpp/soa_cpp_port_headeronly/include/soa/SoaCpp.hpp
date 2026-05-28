#pragma once

#include <Eigen/Eigen>
#include <algorithm>
#include <cmath>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace soa {

using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using Vec6 = Eigen::Matrix<double, 6, 1>;

inline Mat zeros(int r, int c) { return Mat::Zero(r, c); }
inline Vec zeros(int n) { return Vec::Zero(n); }
inline Mat eye(int n) { return Mat::Identity(n, n); }

inline Mat vstack(const std::vector<Mat>& blocks) {
    if (blocks.empty()) return Mat(0, 0);
    int cols = blocks.front().cols();
    int rows = 0;
    for (const auto& b : blocks) {
        if (b.cols() != cols) throw std::runtime_error("vstack: column mismatch");
        rows += static_cast<int>(b.rows());
    }
    Mat out(rows, cols);
    int r = 0;
    for (const auto& b : blocks) {
        out.block(r, 0, b.rows(), b.cols()) = b;
        r += static_cast<int>(b.rows());
    }
    return out;
}

inline Vec vstackVec(const std::vector<Vec>& blocks) {
    int rows = 0;
    for (const auto& b : blocks) rows += static_cast<int>(b.size());
    Vec out(rows);
    int r = 0;
    for (const auto& b : blocks) {
        out.segment(r, b.size()) = b;
        r += static_cast<int>(b.size());
    }
    return out;
}

inline Mat hstack(const std::vector<Mat>& blocks) {
    if (blocks.empty()) return Mat(0, 0);
    int rows = blocks.front().rows();
    int cols = 0;
    for (const auto& b : blocks) {
        if (b.rows() != rows) throw std::runtime_error("hstack: row mismatch");
        cols += static_cast<int>(b.cols());
    }
    Mat out(rows, cols);
    int c = 0;
    for (const auto& b : blocks) {
        out.block(0, c, b.rows(), b.cols()) = b;
        c += static_cast<int>(b.cols());
    }
    return out;
}

inline Mat blockDiag(const std::vector<Mat>& blocks) {
    int rows = 0, cols = 0;
    for (const auto& b : blocks) { rows += b.rows(); cols += b.cols(); }
    Mat out = Mat::Zero(rows, cols);
    int r = 0, c = 0;
    for (const auto& b : blocks) {
        out.block(r, c, b.rows(), b.cols()) = b;
        r += static_cast<int>(b.rows());
        c += static_cast<int>(b.cols());
    }
    return out;
}

inline Mat selectRowsCols(const Mat& A, const std::vector<int>& rows, const std::vector<int>& cols) {
    Mat out(rows.size(), cols.size());
    for (int i = 0; i < static_cast<int>(rows.size()); ++i)
        for (int j = 0; j < static_cast<int>(cols.size()); ++j)
            out(i, j) = A(rows[i], cols[j]);
    return out;
}

inline Mat3 skew3(const Vec3& v) {
    Mat3 S;
    S << 0.0, -v(2), v(1),
         v(2), 0.0, -v(0),
        -v(1), v(0), 0.0;
    return S;
}

inline Mat skew(const Vec& v) {
    if (v.size() < 3) throw std::runtime_error("skew requires at least 3 entries");
    return skew3(v.head<3>());
}

// Spatial force transform for a point offset r: [m; f] -> [m + r x f; f].
inline Mat6 phi(const Vec3& r) {
    Mat6 P = Mat6::Zero();
    P.block<3,3>(0,0) = Mat3::Identity();
    P.block<3,3>(0,3) = skew3(r);
    P.block<3,3>(3,3) = Mat3::Identity();
    return P;
}

inline Mat6 getR6(const Mat3& R) {
    Mat6 R6 = Mat6::Zero();
    R6.block<3,3>(0,0) = R;
    R6.block<3,3>(3,3) = R;
    return R6;
}

inline Mat3 q2R(const Vec& q4) {
    if (q4.size() != 4) throw std::runtime_error("q2R expects [x,y,z,w]");
    Eigen::Quaterniond q(q4(3), q4(0), q4(1), q4(2));
    q.normalize();
    return q.toRotationMatrix();
}

inline Mat3 rotvecToMatrix(const Vec3& rv) {
    double angle = rv.norm();
    if (angle < 1e-14) return Mat3::Identity();
    return Eigen::AngleAxisd(angle, rv / angle).toRotationMatrix();
}

inline Mat hingeMap(const std::string& type) {
    if (type == "revx") { Mat H(1, 6); H << 1,0,0,0,0,0; return H; }
    if (type == "revy") { Mat H(1, 6); H << 0,1,0,0,0,0; return H; }
    if (type == "revz") { Mat H(1, 6); H << 0,0,1,0,0,0; return H; }
    if (type == "spherical") {
        Mat H = Mat::Zero(3, 6);
        H.block(0, 0, 3, 3) = Mat3::Identity();
        return H;
    }
    if (type == "free") return Mat::Identity(6, 6);
    if (type == "fixed") return Mat(0, 6);
    throw std::runtime_error("unknown joint type: " + type);
}

inline Vec quatDerivative(const Vec& theta, const Vec& beta) {
    if (theta.size() < 4 || beta.size() < 3) throw std::runtime_error("quatDerivative size error");
    double x = theta(0), y = theta(1), z = theta(2), w = theta(3);
    double wx = beta(0), wy = beta(1), wz = beta(2);
    Vec qdot(4);
    qdot << 0.5 * ( w*wx + y*wz - z*wy),
            0.5 * ( w*wy + z*wx - x*wz),
            0.5 * ( w*wz + x*wy - y*wx),
           -0.5 * ( x*wx + y*wy + z*wz);
    return qdot;
}

inline Mat crm(const Vec6& V) {
    const Vec3 w = V.segment<3>(0);
    const Vec3 v = V.segment<3>(3);
    Mat6 C = Mat6::Zero();
    C.block<3,3>(0,0) = skew3(w);
    C.block<3,3>(3,0) = skew3(v);
    C.block<3,3>(3,3) = skew3(w);
    return C;
}

inline Mat crf(const Vec6& V) {
    return -crm(V).transpose();
}

inline Mat getA(const Mat& PI_end, const Vec3& klOO) {
    // Equivalent to Python SOALIB.get_A(PI_end, klOO): stack modal end block and rigid shift.
    return vstack({PI_end.transpose(), phi(klOO)});
}

inline Mat solveLinear(const Mat& A, const Mat& B) {
    if (A.rows() == 0 || A.cols() == 0) return Mat::Zero(A.cols(), B.cols());
    return A.colPivHouseholderQr().solve(B);
}

inline Vec solveLinearVec(const Mat& A, const Vec& b) {
    if (A.rows() == 0 || A.cols() == 0) return Vec::Zero(A.cols());
    return A.colPivHouseholderQr().solve(b);
}

struct Joint {
    std::string type;
    Mat H;
    Vec3 klOO;
    Vec3 klOC;
    Vec3 klOO_u;
    double L{};

    Joint() = default;
    Joint(const Vec3& klOO_, std::string H_type) : type(std::move(H_type)), H(hingeMap(type)), klOO(klOO_) {
        L = klOO.norm();
        klOC = klOO / 2.0;
        klOO_u = (L > 0.0) ? (klOO / L).eval() : Vec3(Vec3::UnitX());
    }

    int thetaSize() const {
        if (type == "revx" || type == "revy" || type == "revz") return 1;
        if (type == "spherical") return 4;
        if (type == "free") return 7;
        if (type == "fixed") return 0;
        throw std::runtime_error("unknown joint type: " + type);
    }
    int betaSize() const {
        if (type == "revx" || type == "revy" || type == "revz") return 1;
        if (type == "spherical") return 3;
        if (type == "free") return 6;
        if (type == "fixed") return 0;
        throw std::runtime_error("unknown joint type: " + type);
    }

    std::pair<Vec, Vec> theta2X(const Vec& theta) const {
        Vec q(4);
        Vec trans = klOO;
        if (type == "revx") { double a = theta(0); q << std::sin(a/2), 0, 0, std::cos(a/2); }
        else if (type == "revy") { double a = theta(0); q << 0, std::sin(a/2), 0, std::cos(a/2); }
        else if (type == "revz") { double a = theta(0); q << 0, 0, std::sin(a/2), std::cos(a/2); }
        else if (type == "spherical") { q = theta.head(4); }
        else if (type == "free") { q = theta.head(4); trans = theta.segment(4, 3); }
        else if (type == "fixed") { q << 0, 0, 0, 1; }
        else throw std::runtime_error("unknown joint type: " + type);
        Vec X(7);
        X << q, trans;
        return {X, q};
    }

    Vec thetaDot(const Vec& theta, const Vec& beta) const {
        if (type.rfind("rev", 0) == 0) return beta;
        if (type == "spherical") return quatDerivative(theta, beta);
        if (type == "free") {
            Vec out(7);
            out << quatDerivative(theta.head(4), beta.head(3)), beta.segment(3, 3);
            return out;
        }
        if (type == "fixed") return Vec(0);
        throw std::runtime_error("unknown joint type: " + type);
    }
};

struct RigidProperties {
    double rho{};
    double w{};
    double h{};
    double A{};
    double L{};
    Vec3 CkJk = Vec3::Zero();
    Mat Mk;

    RigidProperties() = default;
    RigidProperties(double rho_, double w_, double h_) : rho(rho_), w(w_), h(h_) {}

    Mat getMk(double m, const Vec3& CkJk_, const Vec3& klOC) const {
        Mat6 MC = Mat6::Zero();
        MC.block<3,3>(0,0) = CkJk_.asDiagonal();
        MC.block<3,3>(3,3) = m * Mat3::Identity();
        Mat6 P = phi(klOC);
        return P * MC * P.transpose();
    }
};

struct ModeLabel {
    int mode{};
    double freq_hz{};
    std::string label;
    double score{};
};

struct FlexProperties {
    double E{};
    double G{};
    double c{};
    int n_nd{};
    int n_md{};
    int n_elem{};
    double L_elem{};
    double F_axial{};
    bool constant_axial_load{false};
    bool has_mode_selection{false};
    std::map<std::string, int> mode_selection;
    std::vector<ModeLabel> modes;
    std::vector<Vec3> klOO_nd;

    Mat PI, PI_e, PI_end;
    Vec omega2, omega;
    Mat K_fl, M_fl, C_fl;
    Vec3 p_0 = Vec3::Zero();
    Mat3 J_0 = Mat3::Zero();
    Mat J_1, S_1, F_1;
    Mat L_fl, U_fl, D_fl;

    FlexProperties() = default;
    FlexProperties(double E_, double G_, double c_, int n_nd_, int n_md_)
        : E(E_), G(G_), c(c_), n_nd(n_nd_), n_md(n_md_), n_elem(n_nd_ - 1) {}

    void setModeSelection(std::map<std::string, int> selection) {
        mode_selection = std::move(selection);
        has_mode_selection = true;
    }

    void setConstantAxialLoad(double F_axial_) {
        F_axial = F_axial_;
        constant_axial_load = true;
    }
};

struct SystemState {
    std::vector<Vec> Theta;
    std::vector<Vec> Beta;
    std::vector<Vec> Eta;
    std::vector<Vec> Eta_dot;

    SystemState() = default;
    SystemState(std::vector<Vec> theta, std::vector<Vec> beta, std::vector<Vec> eta, std::vector<Vec> eta_dot)
        : Theta(std::move(theta)), Beta(std::move(beta)), Eta(std::move(eta)), Eta_dot(std::move(eta_dot)) {}

    Vec pack() const {
        std::vector<Vec> blocks;
        blocks.insert(blocks.end(), Theta.begin(), Theta.end());
        blocks.insert(blocks.end(), Beta.begin(), Beta.end());
        blocks.insert(blocks.end(), Eta.begin(), Eta.end());
        blocks.insert(blocks.end(), Eta_dot.begin(), Eta_dot.end());
        return vstackVec(blocks);
    }

    static SystemState unpack(const Vec& S, const std::vector<Joint*>& joints, const std::vector<FlexProperties*>& flexs) {
        std::vector<Vec> theta, beta, eta, eta_dot;
        int idx = 0;
        for (auto* j : joints) { int sz = j->thetaSize(); theta.push_back(S.segment(idx, sz)); idx += sz; }
        for (auto* j : joints) { int sz = j->betaSize(); beta.push_back(S.segment(idx, sz)); idx += sz; }
        for (auto* f : flexs) { int sz = f->n_md; eta.push_back(S.segment(idx, sz)); idx += sz; }
        for (auto* f : flexs) { int sz = f->n_md; eta_dot.push_back(S.segment(idx, sz)); idx += sz; }
        return SystemState(theta, beta, eta, eta_dot);
    }
};

class StructuralAnalysisBDRect {
public:
    Vec3 klOO, klOC;
    double w{}, h{}, L{}, A{}, rho{}, m{}, E{}, G{}, c{};
    int n_nd{}, n_md{}, n_elem{}, n_md_compute{};
    double L_elem{}, m_e{};
    bool constant_axial_load{false};
    double F_axial{0.0};
    bool has_mode_selection{false};
    std::map<std::string, int> mode_selection;

    Mat K_st, M_st, PI, PI_e, K_fl, M_fl, C_fl;
    Vec omega2, omega, m_nd;
    std::vector<Mat3> J_list;
    std::vector<Vec3> p_list;
    Mat lambda_, gamma;
    std::vector<ModeLabel> modes;
    Vec3 p_0 = Vec3::Zero();
    Mat3 J_0 = Mat3::Zero();
    Mat J_1, S_1, F_1, F_0, G_0, E_0;

    StructuralAnalysisBDRect(const Joint& joint, const RigidProperties& rigid, const FlexProperties& flex) {
        klOO = joint.klOO;
        klOC = joint.klOC;
        w = rigid.w; h = rigid.h; L = rigid.L; A = rigid.A; rho = rigid.rho;
        m = rho * A * L;
        E = flex.E; G = flex.G; c = flex.c;
        n_nd = flex.n_nd; n_md = flex.n_md; n_elem = n_nd - 1;
        L_elem = L / static_cast<double>(n_elem);
        m_e = rho * A * L_elem;
        constant_axial_load = flex.constant_axial_load;
        F_axial = flex.F_axial;
        n_md_compute = flex.n_md;
        has_mode_selection = flex.has_mode_selection;
        mode_selection = flex.mode_selection;

        K_st = getKst();
        M_st = getMst();
        PI = getPI();
        K_fl = getKfl();
        M_fl = getMfl();
        C_fl = getCfl();
    }

private:
    Mat getStiffMatRect3D() const {
        double a = w / 2.0;
        double b = h / 2.0;
        double Le = L_elem;
        double k_y = 1.2;
        double k_z = k_y;
        double K = a * std::pow(b, 3) * (16.0/3.0 - 3.36 * a / b * (1.0 - std::pow(a, 4) / (12.0 * std::pow(b, 4))));
        double I_y = w * std::pow(h, 3) / 12.0;
        double I_z = h * std::pow(w, 3) / 12.0;
        double phi_y = 12.0 * E * I_z * k_y / (A * G * Le * Le);
        double phi_z = 12.0 * E * I_y * k_z / (A * G * Le * Le);
        double S = G * K / Le;
        double X = A * E / Le;
        double Y1 = 12.0 * E * I_z / ((1.0 + phi_y) * std::pow(Le, 3));
        double Y2 = 6.0 * E * I_z / ((1.0 + phi_y) * Le * Le);
        double Y3 = (4.0 + phi_y) * E * I_z / ((1.0 + phi_y) * Le);
        double Y4 = (2.0 - phi_y) * E * I_z / ((1.0 + phi_y) * Le);
        double Z1 = 12.0 * E * I_y / ((1.0 + phi_z) * std::pow(Le, 3));
        double Z2 = 6.0 * E * I_y / ((1.0 + phi_z) * Le * Le);
        double Z3 = (4.0 + phi_z) * E * I_y / ((1.0 + phi_z) * Le);
        double Z4 = (2.0 - phi_z) * E * I_y / ((1.0 + phi_z) * Le);

        Mat k = Mat::Zero(12, 12);
        std::vector<double> d0{X,Y1,Z1,S,Z3,Y3,X,Y1,Z1,S,Z3,Y3};
        for (int i = 0; i < 12; ++i) k(i, i) = d0[i];
        auto addDiag = [&](const std::vector<double>& d, int off) {
            for (int i = 0; i < static_cast<int>(d.size()); ++i) {
                if (i + off < 12) k(i + off, i) += d[i];
                if (i + off < 12) k(i, i + off) += d[i];
            }
        };
        addDiag({0,0,-Z2,0,0,-Y2,0,0,Z2,0}, 2);
        addDiag({0,Y2,0,0,Z2,0,0,-Y2}, 4);
        addDiag({-X,-Y1,-Z1,-S,Z4,Y4}, 6);
        addDiag({0,0,-Z2,0}, 8);
        addDiag({0,Y2}, 10);

        std::vector<int> perm{3,4,5,0,1,2,9,10,11,6,7,8};
        Mat k_perm = selectRowsCols(k, perm, perm);
        Mat k_geo = Mat::Zero(12, 12);
        if (constant_axial_load) {
            double P = F_axial;
            Mat ksig(4,4);
            ksig << 36, 3*Le, -36, 3*Le,
                    3*Le, 4*Le*Le, -3*Le, -Le*Le,
                    -36, -3*Le, 36, -3*Le,
                    3*Le, -Le*Le, -3*Le, 4*Le*Le;
            ksig *= P / (30.0 * Le);
            std::vector<int> dof_xy{4,2,10,8};
            std::vector<int> dof_xz{5,1,11,7};
            for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
                k_geo(dof_xy[i], dof_xy[j]) += ksig(i,j);
                k_geo(dof_xz[i], dof_xz[j]) += ksig(i,j);
            }
        }
        return k_perm + k_geo;
    }

    Mat getKst() const {
        Mat k = getStiffMatRect3D();
        Mat K = Mat::Zero(6*n_nd, 6*n_nd);
        for (int i = 0; i < n_nd - 1; ++i) K.block(6*i, 6*i, 12, 12) += k;
        return K;
    }

    Mat getMst() {
        m_nd = Vec::Constant(n_nd, m_e);
        m_nd(0) = m_e / 2.0;
        m_nd(n_nd - 1) = m_e / 2.0;
        std::vector<Mat> blocks;
        J_list.resize(n_nd);
        p_list.resize(n_nd);
        for (int i = 0; i < n_nd; ++i) {
            Vec3 p = Vec3::Zero();
            if (i == 0) p << 0.25 * L_elem, 0, 0;
            else if (i == n_nd - 1) p << -0.25 * L_elem, 0, 0;
            double L_slice = (0 < i && i < n_nd - 1) ? L_elem : L_elem / 2.0;
            Mat3 Jc = Mat3::Zero();
            Jc.diagonal() << w*w + h*h, L_slice*L_slice + h*h, L_slice*L_slice + w*w;
            Jc *= m_nd(i) / 12.0;
            Mat3 J = Jc - m_nd(i) * skew3(p) * skew3(p);
            Mat6 Mj = Mat6::Zero();
            Mj.block<3,3>(0,0) = J;
            Mj.block<3,3>(0,3) = m_nd(i) * skew3(p);
            Mj.block<3,3>(3,0) = -m_nd(i) * skew3(p);
            Mj.block<3,3>(3,3) = m_nd(i) * Mat3::Identity();
            blocks.push_back(Mj);
            J_list[i] = J;
            p_list[i] = p;
        }
        return blockDiag(blocks);
    }

    std::vector<ModeLabel> identifyModeLabels() const {
        std::vector<ModeLabel> labels;
        for (int r = 0; r < PI_e.cols(); ++r) {
            Vec pie = PI_e.col(r);
            double torsion_x = 0, axial_x = 0, bending_xy = 0, bending_xz = 0;
            for (int i = 0; i < n_nd; ++i) {
                torsion_x += std::pow(L * pie(6*i + 0), 2);
                axial_x += std::pow(pie(6*i + 3), 2);
                bending_xy += std::abs(pie(6*i + 4)) + std::abs(L * pie(6*i + 2));
                bending_xz += std::abs(pie(6*i + 5)) + std::abs(L * pie(6*i + 1));
            }
            torsion_x = std::sqrt(torsion_x);
            axial_x = std::sqrt(axial_x);
            std::map<std::string, double> scores{{"torsion_x", torsion_x}, {"axial_x", axial_x}, {"bending_xy", bending_xy}, {"bending_xz", bending_xz}};
            auto it = std::max_element(scores.begin(), scores.end(), [](const auto& a, const auto& b){ return a.second < b.second; });
            constexpr double pi = 3.141592653589793238462643383279502884;
            labels.push_back(ModeLabel{r+1, omega(r) / (2.0*pi), it->first, it->second});
        }
        return labels;
    }

    Mat getPI() {
        if (n_md_compute == 0) {
            PI_e = Mat::Zero(6*n_nd, 0);
            omega2 = Vec(0); omega = Vec(0); modes.clear(); n_md = 0;
            gamma = Mat::Zero(3*n_nd, 0);
            lambda_ = Mat::Zero(3*n_nd, 0);
            return PI_e;
        }
        std::vector<int> dof_int;
        for (int i = 0; i < 6*n_nd; ++i) if (i >= 6) dof_int.push_back(i);
        Mat K_int = selectRowsCols(K_st, dof_int, dof_int);
        Mat M_int = selectRowsCols(M_st, dof_int, dof_int);
        Eigen::GeneralizedSelfAdjointEigenSolver<Mat> es(K_int, M_int);
        if (es.info() != Eigen::Success) throw std::runtime_error("generalized eigen solve failed");
        int cand = std::min(n_md_compute, static_cast<int>(K_int.rows()));
        Vec eig_val = es.eigenvalues().head(cand);
        Mat PI_int = es.eigenvectors().leftCols(cand);
        Mat PI_full = Mat::Zero(6*n_nd, cand);
        PI_full.block(6, 0, 6*n_nd - 6, cand) = PI_int;
        PI_e = PI_full;
        omega2 = eig_val;
        omega = eig_val.cwiseMax(0.0).cwiseSqrt();
        auto all_modes = identifyModeLabels();
        std::vector<int> keep;
        if (has_mode_selection) {
            std::map<std::string, int> used;
            for (const auto& kv : mode_selection) used[kv.first] = 0;
            for (int i = 0; i < static_cast<int>(all_modes.size()); ++i) {
                const auto& label = all_modes[i].label;
                auto wanted = mode_selection.find(label);
                if (wanted != mode_selection.end() && used[label] < wanted->second) {
                    keep.push_back(i);
                    used[label]++;
                }
            }
        } else {
            keep.resize(cand);
            std::iota(keep.begin(), keep.end(), 0);
        }
        if (keep.empty()) throw std::runtime_error("No modes matched the selection criteria");
        Mat PI_keep(6*n_nd, keep.size());
        Vec omega2_keep(keep.size());
        modes.clear();
        for (int i = 0; i < static_cast<int>(keep.size()); ++i) {
            PI_keep.col(i) = PI_full.col(keep[i]);
            omega2_keep(i) = eig_val(keep[i]);
            modes.push_back(all_modes[keep[i]]);
        }
        PI_e = PI_keep;
        omega2 = omega2_keep;
        omega = omega2.cwiseMax(0.0).cwiseSqrt();
        n_md = static_cast<int>(keep.size());
        gamma = Mat::Zero(3*n_nd, n_md);
        lambda_ = Mat::Zero(3*n_nd, n_md);
        for (int i = 0; i < n_nd; ++i) {
            lambda_.block(3*i, 0, 3, n_md) = PI_e.block(6*i, 0, 3, n_md);
            gamma.block(3*i, 0, 3, n_md) = PI_e.block(6*i + 3, 0, 3, n_md);
        }
        return PI_e;
    }

    Mat getKfl() const {
        Mat K = Mat::Zero(n_md + 6, n_md + 6);
        if (n_md > 0) K.block(0, 0, n_md, n_md) = PI.transpose() * K_st * PI;
        return K;
    }

    void getModalInt() {
        Vec3 p0 = Vec3::Zero();
        Mat p1 = Mat::Zero(3, n_md);
        Mat3 J0 = Mat3::Zero();
        Mat J1 = Mat::Zero(3, 3*n_md);
        Mat F0 = Mat::Zero(3, n_md);
        Mat F1 = Mat::Zero(3*n_md, n_md);
        Mat G0 = Mat::Zero(n_md, n_md);
        Mat E0 = Mat::Zero(3, n_md);
        Mat S1 = Mat::Zero(3, 3*n_md);

        for (int i = 0; i < n_nd; ++i) {
            Vec3 klkO(i * L_elem, 0, 0);
            Mat3 klkO_skew = skew3(klkO);
            Mat3 p_skew = skew3(p_list[i]);
            p0 += m_nd(i) * (p_list[i] + klkO);
            J0 += J_list[i] - m_nd(i) * (klkO_skew * klkO_skew + p_skew * klkO_skew + klkO_skew * p_skew);
            for (int r = 0; r < n_md; ++r) {
                Vec3 gamma_r = gamma.block(3*i, r, 3, 1);
                Vec3 lambda_r = lambda_.block(3*i, r, 3, 1);
                F0.col(r) += J_list[i] * lambda_r + m_nd(i) * (klkO_skew + p_skew) * gamma_r - m_nd(i) * klkO_skew * p_skew * lambda_r;
                E0.col(r) += m_nd(i) * (gamma_r - p_skew * lambda_r);
                p1.col(r) += m_nd(i) * gamma_r;
                J1.block(0, 3*r, 3, 3) += m_nd(i) * skew3(gamma_r) * (klkO_skew + p_skew);
                S1.block(0, 3*r, 3, 3) += skew3(m_nd(i) * p_skew * lambda_r) * klkO_skew - J_list[i] * skew3(lambda_r);
                for (int s = 0; s < n_md; ++s) {
                    Vec3 gamma_s = gamma.block(3*i, s, 3, 1);
                    Vec3 lambda_s = lambda_.block(3*i, s, 3, 1);
                    double term = (lambda_r.transpose() * J_list[i] * lambda_s)(0,0)
                        + m_nd(i) * (
                            (lambda_r.transpose() * p_skew * gamma_s)(0,0)
                          + (lambda_s.transpose() * p_skew * gamma_r)(0,0)
                          + (gamma_r.transpose() * gamma_s)(0,0));
                    G0(r,s) += term;
                }
            }
        }
        p_0 = p0 / m;
        J_0 = J0;
        J_1 = J1;
        F_0 = F0;
        F_1 = F1;
        G_0 = G0;
        E_0 = E0;
        S_1 = S1;
    }

    Mat getMfl() {
        getModalInt();
        Mat3 p0_skew = skew3(p_0);
        Mat rw1 = hstack({G_0, F_0.transpose(), E_0.transpose()});
        Mat rw2 = hstack({F_0, J_0, m * p0_skew});
        Mat rw3 = hstack({E_0, -m * p0_skew, m * Mat3::Identity()});
        return vstack({rw1, rw2, rw3});
    }

    Mat getCfl() const {
        Mat C = Mat::Zero(M_fl.rows(), M_fl.cols());
        if (n_md > 0) C.block(0, 0, n_md, n_md) = (c * omega2.array()).matrix().asDiagonal();
        return C;
    }
};

class SOABody {
public:
    struct ExternalLoad {
        int node{-1};
        std::function<Vec6(double, const SystemState&)> fun;
    };
    struct ZSpring { int node{}; double k{}; double c{}; };
    struct TrackTensioner { int node{}; double F_TT_z0{}; double k_TT{}; double c_TT{}; };
    struct EarthModel { double k_c{}, k_phi{}, n{}, c{}, c_soil{}, mu_soil{}, b{}; };
    struct Force {
        Vec tau;
        std::vector<ExternalLoad> F_ext;
        double k_TS{0.0}, c_TS{0.0}, theta0_TS{0.0};
        std::vector<ZSpring> z_springs;
        std::vector<TrackTensioner> track_tensioners;
        Vec6 F_axial_global = Vec6::Zero();
        bool F_axial_track{false};
        bool has_earth_model{false};
        EarthModel earth_model;
    };
    struct InitialCondition {
        Vec theta0, beta0, eta0, eta_dot0;
        InitialCondition() = default;
        InitialCondition(const Joint& joint, const FlexProperties& flex) {
            theta0 = Vec::Zero(joint.thetaSize());
            if (theta0.size() == 4) theta0(theta0.size() - 1) = 1.0;
            else if (theta0.size() == 7) theta0(3) = 1.0;
            beta0 = Vec::Zero(joint.betaSize());
            eta0 = Vec::Zero(flex.n_md);
            eta_dot0 = Vec::Zero(flex.n_md);
        }
    };

    Joint joint;
    RigidProperties rigid;
    FlexProperties flex;
    Force force;
    InitialCondition initialcondition;
    double m{};

    SOABody() = default;
    SOABody(const Joint& joint_, const RigidProperties& rigid_, const FlexProperties& flex_) : joint(joint_), rigid(rigid_), flex(flex_) {
        force.tau = Vec::Zero(joint.betaSize());
        rigid.A = rigid.h * rigid.w;
        rigid.L = joint.L;
        flex.L_elem = joint.L / static_cast<double>(flex.n_elem);
        flex.klOO_nd.resize(flex.n_nd);
        for (int j = 0; j < flex.n_nd; ++j) flex.klOO_nd[j] = static_cast<double>(j) * (joint.klOO / static_cast<double>(flex.n_elem));
        m = rigid.rho * rigid.A * joint.L;
        rigid.CkJk << (1.0/12.0) * m * (rigid.h*rigid.h + rigid.w*rigid.w),
                       (1.0/12.0) * m * (rigid.h*rigid.h + joint.L*joint.L),
                       (1.0/12.0) * m * (rigid.w*rigid.w + joint.L*joint.L);
        rigid.Mk = rigid.getMk(m, rigid.CkJk, joint.klOC);
        joint.klOO_u = (joint.L > 0.0) ? (joint.klOO / joint.L).eval() : Vec3(Vec3::UnitX());

        StructuralAnalysisBDRect body_analysis(joint, rigid, flex);
        getBodyAnalysisRot(body_analysis);
        initialcondition = InitialCondition(joint, flex);
        getDmOffline();
    }

    void setTau(const Vec& tau) { force.tau = tau; }
    void setFext(int node, const Vec6& F_ext) {
        ExternalLoad load;
        load.node = node;
        load.fun = [F_ext](double, const SystemState&) { return F_ext; };
        force.F_ext.push_back(load);
    }
    void setFext(const Vec6& F_ext) { setFext(-1, F_ext); }
    void setFextFunction(int node, std::function<Vec6(double, const SystemState&)> fun) {
        force.F_ext.push_back(ExternalLoad{node, std::move(fun)});
    }
    void setTS(double k_TS, double c_TS, double theta0_TS) { force.k_TS = k_TS; force.c_TS = c_TS; force.theta0_TS = theta0_TS; }
    void setImpulseForce(double ts, double dt, const Vec6& F_impulse, int node = -1) {
        setFextFunction(node, [=](double t, const SystemState&) {
            if (ts <= t && t <= ts + dt) return Vec6(F_impulse);
            return Vec6(Vec6::Zero());
        });
    }
    void setZSpring(double k_z, double c_z, int node) { force.z_springs.push_back(ZSpring{node, k_z, c_z}); }
    void setGlobalAxialForce(double F_axial) { force.F_axial_global << 0,0,0,F_axial,0,0; force.F_axial_track = true; }
    void setTrackTensioner(double F_TT_z0, double c_TT, double z_0, int node) { force.track_tensioners.push_back(TrackTensioner{node, F_TT_z0, F_TT_z0 / z_0, c_TT}); }
    void setEarthModel(double c, const std::string& soil_type = "Soft soil") {
        static const std::map<std::string, std::tuple<double,double,double,double,double>> presets = {
            {"Hard dirt", {2.0e4, 6.0e5, 0.7, 5.0e2, 0.058}},
            {"Soft soil", {1.2e4, 3.5e5, 1.2, 1.0e2, 0.036}},
            {"Sand",      {0.8e4, 2.0e5, 1.6, 0.0,   0.070}},
            {"Mud",       {0.4e4, 0.8e5, 2.0, 0.5e2, 0.018}}
        };
        auto it = presets.find(soil_type);
        if (it == presets.end()) it = presets.find("Soft soil");
        auto [k_c, k_phi, n, c_soil, mu_soil] = it->second;
        force.earth_model = EarthModel{k_c, k_phi, n, c, c_soil, mu_soil, rigid.w};
        force.has_earth_model = true;
    }
    void setEarthModelManual(double c, double k_c, double k_phi, double n, double c_soil, double mu_soil) {
        force.earth_model = EarthModel{k_c, k_phi, n, c, c_soil, mu_soil, rigid.w};
        force.has_earth_model = true;
    }

    Vec3 nodePosition(int node) const {
        int idx = node;
        if (idx < 0) idx = flex.n_nd + idx;
        if (idx < 0 || idx >= flex.n_nd) throw std::out_of_range("node index out of range");
        double s = static_cast<double>(idx) / static_cast<double>(flex.n_nd - 1);
        return s * joint.klOO;
    }

    Vec getFextTerm(const SystemState& state, double t) const {
        int n_md = flex.n_md;
        Vec F_term = Vec::Zero(n_md + 6);
        for (const auto& load : force.F_ext) {
            int node = load.node;
            if (node < 0) node = flex.n_nd + node;
            Vec6 F_j = load.fun(t, state);
            Mat PI_j = flex.PI.block(6*node, 0, 6, n_md);
            Vec3 r_j = nodePosition(node);
            Vec modal = PI_j.transpose() * F_j;
            Vec6 rigidTerm = phi(r_j) * F_j;
            F_term += vstackVec({modal, rigidTerm});
        }
        return F_term;
    }

    Vec getTSTerm(const Vec& theta, const Vec& beta) const {
        if (joint.type.rfind("rev", 0) == 0) return -force.k_TS * (theta.array() - force.theta0_TS).matrix() - force.c_TS * beta;
        if (joint.type == "spherical") return Vec::Zero(3);
        if (joint.type == "fixed") return Vec(0);
        if (joint.type == "free") return Vec::Zero(6);
        throw std::runtime_error("unknown joint type: " + joint.type);
    }

    std::tuple<std::vector<Vec3>, std::vector<Vec3>, Mat3> getTrackKin(
        const Vec3& last_end, const Vec3& last_end_dot, const Mat3& R_i_in,
        const Mat3& R3, const Vec6& V, const Vec& eta, const Vec& eta_dot) const {
        Mat3 R_i = R_i_in * R3;
        std::vector<Vec3> nodes_pos, nodes_V;
        for (int j = 0; j < flex.n_nd; ++j) {
            Vec3 pos_und = flex.klOO_nd[j];
            Vec3 u_j = flex.PI.block(6*j + 3, 0, 3, flex.n_md) * eta;
            Vec3 pos_glob = last_end + R_i * (pos_und + u_j);
            nodes_pos.push_back(pos_glob);
            Vec6 V_und6 = phi(flex.klOO_nd[j]).transpose() * V;
            Vec3 V_und = V_und6.segment<3>(3);
            Vec3 u_j_dot = flex.PI.block(6*j + 3, 0, 3, flex.n_md) * eta_dot;
            Vec3 V_glob = last_end_dot + R_i * (V_und + u_j_dot);
            nodes_V.push_back(V_glob);
        }
        return {nodes_pos, nodes_V, R_i};
    }

    Vec getGlobalForcesTerm(const std::vector<Vec3>& pos, const std::vector<Vec3>& pos_dot, const Mat3& R_i) const {
        int n_md = flex.n_md;
        Vec F_term = Vec::Zero(n_md + 6);
        Mat6 R6 = getR6(R_i);
        for (const auto& spring : force.z_springs) {
            int idx = spring.node < 0 ? flex.n_nd + spring.node : spring.node;
            double F_z = -spring.k * pos[idx](2) - spring.c * pos_dot[idx](2);
            Vec6 F_glob; F_glob << 0,0,0,0,0,F_z;
            Vec6 F_loc = R6.transpose() * F_glob;
            Mat PI_j = flex.PI.block(6*idx, 0, 6, n_md);
            F_term += vstackVec({PI_j.transpose() * F_loc, phi(nodePosition(idx)) * F_loc});
        }
        for (const auto& tt : force.track_tensioners) {
            int idx = tt.node < 0 ? flex.n_nd + tt.node : tt.node;
            double z = pos[idx](2);
            double z_dot = pos_dot[idx](2);
            double F_TT_mag = tt.F_TT_z0 - tt.k_TT * z - tt.c_TT * z_dot;
            Vec6 F_glob; F_glob << 0,0,0,0,0,-F_TT_mag;
            Vec6 F_loc = R6.transpose() * F_glob;
            Mat PI_j = flex.PI.block(6*idx, 0, 6, n_md);
            F_term += vstackVec({PI_j.transpose() * F_loc, phi(nodePosition(idx)) * F_loc});
        }
        if (force.has_earth_model) {
            const auto& em = force.earth_model;
            for (int idx = 0; idx < flex.n_nd; ++idx) {
                double z = pos[idx](2), z_dot = pos_dot[idx](2);
                if (z < 0.0) {
                    double p = -z, p_dot = -z_dot;
                    double A_node = (idx == 0 || idx == flex.n_nd - 1) ? (rigid.w * flex.L_elem) / 2.0 : rigid.w * flex.L_elem;
                    double A_x = (idx == 0 || idx == flex.n_nd - 1) ? 0.5 : 1.0;
                    double F_z_mag = A_node * (em.k_c / em.b + em.k_phi) * std::pow(p, em.n) + em.c * p_dot;
                    F_z_mag = std::max(0.0, F_z_mag);
                    double F_x = -A_x * (1e5 / 2.0) / 30.0;
                    if (F_z_mag > 0.0) {
                        Vec6 F_glob; F_glob << 0,0,0,F_x,0,F_z_mag;
                        Vec6 F_loc = R6.transpose() * F_glob;
                        Mat PI_j = flex.PI.block(6*idx, 0, 6, n_md);
                        F_term += vstackVec({PI_j.transpose() * F_loc, phi(nodePosition(idx)) * F_loc});
                    }
                }
            }
        }
        if (force.F_axial_track) {
            Vec6 F_loc = R6.transpose() * force.F_axial_global;
            Mat PI_end = flex.PI.block(6*(flex.n_nd - 1), 0, 6, n_md);
            F_term += vstackVec({PI_end.transpose() * F_loc, phi(joint.klOO) * F_loc});
        }
        return F_term;
    }

    Vec coriolis(const Vec6& V, const Vec& beta, const Mat& H, int n_md) const {
        Vec6 deltaV = H.transpose() * beta;
        Vec b_eta = Vec::Zero(n_md);
        Vec6 b_r = crm(V) * deltaV - crf(deltaV) * deltaV;
        return vstackVec({b_eta, b_r});
    }

    Vec gyroscopic(const Vec6& V, const Mat& M) const { return crf(V) * M * V; }

    Vec6 coriolisBD(const Vec6& V_k, const Vec6& V_p, const Vec& beta, const Mat& H, const Vec3& klOO, const Mat3& R3) const {
        Vec6 deltaV = H.transpose() * beta;
        Vec3 a01 = skew3(V_k.segment<3>(0)) * deltaV.segment<3>(0);
        Vec3 vp = R3.transpose() * V_p.segment<3>(0);
        Vec3 a02 = skew3(vp) * skew3(vp) * klOO;
        Vec6 out; out << a01, a02;
        return out;
    }

    Vec gyroscopicBD(const Vec6& V_r) const {
        int n_md = flex.n_md;
        Vec out = Vec::Zero(n_md + 6);
        Vec3 omega = V_r.segment<3>(0);
        for (int i = 0; i < n_md; ++i) {
            Mat3 A = flex.S_1.block(0, 3*i, 3, 3) + flex.J_1.block(0, 3*i, 3, 3);
            out(i) = -(omega.transpose() * A * omega)(0,0);
        }
        Mat3 omega_skew = skew3(omega);
        out.segment(n_md, 3) = omega_skew * flex.J_0 * omega;
        out.segment(n_md + 3, 3) = m * omega_skew * omega_skew * flex.p_0;
        return out;
    }

    void setInitialTheta0(const Vec& x) { initialcondition.theta0 = x; }
    void setInitialBeta0(const Vec& x) { initialcondition.beta0 = x; }
    void setInitialEta0(const Vec& x) { initialcondition.eta0 = x; }
    void setInitialEtaDot0(const Vec& x) { initialcondition.eta_dot0 = x; }

    Mat3 getRInitial() const {
        Vec3 u_hat = joint.klOO_u;
        Vec3 i_hat = Vec3::UnitX();
        if ((u_hat - i_hat).norm() < 1e-12) return Mat3::Identity();
        if ((u_hat + i_hat).norm() < 1e-12) { Mat3 R = Mat3::Identity(); R(0,0) = -1; R(1,1) = -1; return R; }
        Vec3 v = i_hat.cross(u_hat);
        double cc = i_hat.dot(u_hat);
        Mat3 V = skew3(v);
        return Mat3::Identity() + V + (V * V) * (1.0 / (1.0 + cc));
    }

    void getBodyAnalysisRot(const StructuralAnalysisBDRect& ba) {
        Mat3 R = getRInitial();
        Mat6 R6 = getR6(R);
        std::vector<Mat> nodeBlocks(flex.n_nd, R6);
        Mat R_nodes = blockDiag(nodeBlocks);
        Mat R_modal = blockDiag({Mat::Identity(ba.n_md, ba.n_md), Mat(R6)});
        flex.PI = R_nodes * ba.PI;
        flex.PI_e = R_nodes * ba.PI_e;
        flex.PI_end = R6 * ba.PI.block(6*(ba.n_nd-1), 0, 6, ba.n_md);
        flex.omega2 = ba.omega2;
        flex.omega = ba.omega;
        flex.modes = ba.modes;
        flex.n_md = ba.n_md;
        flex.K_fl = R_modal * ba.K_fl * R_modal.transpose();
        flex.M_fl = R_modal * ba.M_fl * R_modal.transpose();
        flex.C_fl = R_modal * ba.C_fl * R_modal.transpose();
        flex.p_0 = R * ba.p_0;
        flex.J_0 = R * ba.J_0 * R.transpose();
        flex.S_1 = Mat::Zero(ba.S_1.rows(), ba.S_1.cols());
        flex.J_1 = Mat::Zero(ba.J_1.rows(), ba.J_1.cols());
        flex.F_1 = Mat::Zero(ba.F_1.rows(), ba.F_1.cols());
        for (int i = 0; i < flex.n_md; ++i) {
            flex.S_1.block(0, 3*i, 3, 3) = R * ba.S_1.block(0, 3*i, 3, 3) * R.transpose();
            flex.J_1.block(0, 3*i, 3, 3) = R * ba.J_1.block(0, 3*i, 3, 3) * R.transpose();
            flex.F_1.block(3*i, 0, 3, ba.F_1.cols()) = R * ba.F_1.block(3*i, 0, 3, ba.F_1.cols());
        }
    }

    void getDmOffline() {
        if (flex.n_md == 0) {
            flex.L_fl = Mat::Zero(0,0); flex.U_fl = Mat::Zero(0,6); flex.D_fl = Mat::Zero(6,6);
        } else {
            Mat H_M_fl = hstack({Mat::Identity(flex.n_md, flex.n_md), Mat::Zero(flex.n_md, 6)});
            Mat A_fl = getA(flex.PI_end, joint.klOO);
            Mat Dm = H_M_fl * flex.M_fl * H_M_fl.transpose();
            flex.L_fl = Dm.inverse();
            Mat zeta = H_M_fl * A_fl;
            flex.U_fl = flex.L_fl * zeta;
            flex.D_fl = zeta.transpose() * flex.U_fl;
        }
    }

    Mat getDmInv(const Mat& Gamma, const std::string& type) const {
        if (flex.n_md == 0) return Mat::Zero(0,0);
        if (type == "tip") return flex.L_fl;
        if (type == "not_tip") {
            Mat A = Mat::Identity(6,6) + Gamma * flex.D_fl;
            return flex.L_fl - flex.U_fl * solveLinear(A, Gamma) * flex.U_fl.transpose();
        }
        throw std::runtime_error("unknown D_m type");
    }
};

class ATBIFlex {
public:
    struct Kinematics {
        std::vector<Vec> X, V, a_fl, b_fl;
        std::vector<Mat3> R3_list, R_i;
        std::vector<std::vector<Vec3>> pos, pos_dot;
    };
    struct GatherResult {
        std::vector<Mat> G_pr, g_fl, P_pr_plus;
        std::vector<Vec> nu_pr, nu_m, z_pr_plus;
    };
    struct ScatterResult {
        std::vector<Vec> theta_ddot, eta_ddot, alpha_fl, F_int;
    };

    std::vector<SOABody*> bodies;
    int n{};
    std::vector<Mat> A_fl;
    Vec6 g = (Vec6() << 0,0,0,0,0,9.81).finished();

    ATBIFlex() = default;
    explicit ATBIFlex(const std::vector<SOABody*>& bodies_) : bodies(bodies_), n(static_cast<int>(bodies_.size())) {
        A_fl.resize(n);
        for (int k = 0; k < n; ++k) A_fl[k] = getA(bodies[k]->flex.PI_end, bodies[k]->joint.klOO);
    }

    Kinematics scatterKinematics(const SystemState& state) const {
        Kinematics out;
        out.X.resize(n); out.V.resize(n); out.a_fl.resize(n); out.b_fl.resize(n);
        out.R3_list.resize(n); out.R_i.resize(n+1); out.pos.resize(n); out.pos_dot.resize(n);
        out.R_i[n] = Mat3::Identity();
        Mat3 R3_n = Mat3::Identity();
        for (int k = n - 1; k >= 0; --k) {
            SOABody& body = *bodies[k];
            const Joint& joint = body.joint;
            const Vec& theta = state.Theta[k];
            const Vec& beta = state.Beta[k];
            const Vec& eta = state.Eta[k];
            const Vec& eta_dot = state.Eta_dot[k];
            const Mat& H = joint.H;
            int n_md = body.flex.n_md;
            auto [Xk, q] = joint.theta2X(theta);
            out.X[k] = Xk;
            Vec V_f = eta_dot;
            Vec6 V_r;
            Vec3 last_end = Vec3::Zero();
            Vec3 last_end_dot = Vec3::Zero();
            Mat3 R3;
            if (k == n - 1) {
                Mat3 R_j = q2R(q);
                R3_n = Mat3::Identity();
                R3 = R3_n * R_j;
                out.R3_list[k] = R3;
                V_r = H.transpose() * beta;
                out.a_fl[k] = body.coriolisBD(V_r, Vec6::Zero(), beta, H, Vec3::Zero(), R3);
            } else {
                Mat3 R3_j = q2R(q);
                R3 = R3_n * R3_j;
                Mat6 R6 = getR6(R3);
                out.R3_list[k] = R3;
                V_r = R6.transpose() * A_fl[k+1].transpose() * out.V[k+1] + H.transpose() * beta;
                Vec3 kl = R3.transpose() * out.X[k+1].segment(4, 3);
                Vec6 Vp_tail = out.V[k+1].tail(6);
                out.a_fl[k] = body.coriolisBD(V_r, Vp_tail, beta, H, kl, R3);
                last_end = out.pos[k+1].back();
                last_end_dot = out.pos_dot[k+1].back();
            }
            out.b_fl[k] = body.gyroscopicBD(V_r);
            out.V[k] = vstackVec({V_f, V_r});
            Vec6 Vk_tail = out.V[k].tail(6);
            auto [p, pd, Ri] = body.getTrackKin(last_end, last_end_dot, out.R_i[k+1], R3, Vk_tail, eta, eta_dot);
            out.pos[k] = std::move(p);
            out.pos_dot[k] = std::move(pd);
            out.R_i[k] = Ri;
            if (n_md > 0) {
                Vec3 R3_n_vec = body.flex.PI_end.block(0, 0, 3, n_md) * eta;
                R3_n = rotvecToMatrix(R3_n_vec);
            } else {
                R3_n = Mat3::Identity();
            }
        }
        return out;
    }

    GatherResult gatherATBI(const SystemState& state, const Kinematics& kin, double t) const {
        GatherResult out;
        out.P_pr_plus.resize(n); out.g_fl.resize(n); out.G_pr.resize(n);
        out.nu_m.resize(n); out.nu_pr.resize(n); out.z_pr_plus.resize(n);
        for (int k = 0; k < n; ++k) {
            SOABody& body = *bodies[k];
            const Mat& H_B = body.joint.H;
            const Vec& tau_pr = body.force.tau;
            const Vec& theta = state.Theta[k];
            const Vec& beta = state.Beta[k];
            const Vec& eta = state.Eta[k];
            const Vec& eta_dot = state.Eta_dot[k];
            const Mat& M_fl = body.flex.M_fl;
            const Mat& K_fl = body.flex.K_fl;
            const Mat& C_fl = body.flex.C_fl;
            int n_md = body.flex.n_md;
            Vec F_ext_term = body.getFextTerm(state, t) + body.getGlobalForcesTerm(kin.pos[k], kin.pos_dot[k], kin.R_i[k]);
            Vec tau_TS_term = body.getTSTerm(theta, beta);
            Mat Gamma_fl;
            Mat P_fl;
            if (k == 0) {
                Gamma_fl = Mat::Zero(0, 6);
                P_fl = M_fl;
            } else {
                Mat6 R6 = getR6(kin.R3_list[k-1]);
                Gamma_fl = R6 * out.P_pr_plus[k-1] * R6.transpose();
                P_fl = A_fl[k] * Gamma_fl * A_fl[k].transpose() + M_fl;
            }
            Mat D_m = P_fl.block(0, 0, n_md, n_md);
            Mat mu_fl = P_fl.block(n_md, 0, 6, n_md);
            Mat D_m_inv = body.getDmInv(Gamma_fl, k == 0 ? "tip" : "not_tip");
            out.g_fl[k] = mu_fl * D_m_inv;
            Mat P_pr = P_fl.block(n_md, n_md, 6, 6) - out.g_fl[k] * mu_fl.transpose();
            Mat D_pr = H_B * P_pr * H_B.transpose();
            Mat G_pr;
            if (H_B.rows() == 0) G_pr = Mat::Zero(6, 0);
            else G_pr = P_pr * solveLinear(D_pr.transpose(), H_B).transpose();
            out.G_pr[k] = G_pr;
            Mat tau_pr_bar = Mat::Identity(6,6) - G_pr * H_B;
            out.P_pr_plus[k] = tau_pr_bar * P_pr;
            Vec eta_aug = vstackVec({eta, Vec::Zero(6)});
            Vec eta_dot_aug = vstackVec({eta_dot, Vec::Zero(6)});
            Vec parent_term = Vec::Zero(n_md + 6);
            if (k != 0) {
                parent_term = (A_fl[k] * getR6(kin.R3_list[k-1]) * out.z_pr_plus[k-1]).eval();
            }
            Vec z = parent_term + kin.b_fl[k] + K_fl * eta_aug - F_ext_term + C_fl * eta_dot_aug;
            Vec eps_m = -z.head(n_md);
            out.nu_m[k] = D_m_inv * eps_m;
            Vec z_pr = z.tail(6) + out.g_fl[k] * eps_m + P_pr * kin.a_fl[k].tail(6);
            Vec eps_pr = tau_pr - H_B * z_pr + tau_TS_term;
            out.nu_pr[k] = solveLinearVec(D_pr, eps_pr);
            out.z_pr_plus[k] = z_pr + G_pr * eps_pr;
        }
        return out;
    }

    ScatterResult scatterATBI(const Kinematics& kin, const GatherResult& gr) const {
        ScatterResult out;
        out.alpha_fl.resize(n); out.theta_ddot.resize(n); out.eta_ddot.resize(n); out.F_int.resize(n);
        for (int k = n - 1; k >= 0; --k) {
            SOABody& body = *bodies[k];
            const Mat& H_B = body.joint.H;
            Mat6 R6 = getR6(kin.R3_list[k]);
            Vec6 alpha_pr_plus;
            if (k == n - 1) {
                alpha_pr_plus = Vec6::Zero();
                Vec6 alpha_base = R6.transpose() * g;
                out.theta_ddot[k] = gr.nu_pr[k] - gr.G_pr[k].transpose() * alpha_base;
                Vec6 afl_tail = kin.a_fl[k].tail(6);
                Vec6 alpha_pr = alpha_base + H_B.transpose() * out.theta_ddot[k] + afl_tail;
                out.eta_ddot[k] = gr.nu_m[k] - gr.g_fl[k].transpose() * alpha_pr;
                out.alpha_fl[k] = vstackVec({out.eta_ddot[k], alpha_pr});
            } else {
                alpha_pr_plus = R6.transpose() * A_fl[k+1].transpose() * out.alpha_fl[k+1];
                out.theta_ddot[k] = gr.nu_pr[k] - gr.G_pr[k].transpose() * alpha_pr_plus;
                Vec6 afl_tail = kin.a_fl[k].tail(6);
                Vec6 alpha_pr = alpha_pr_plus + H_B.transpose() * out.theta_ddot[k] + afl_tail;
                out.eta_ddot[k] = gr.nu_m[k] - gr.g_fl[k].transpose() * alpha_pr;
                out.alpha_fl[k] = vstackVec({out.eta_ddot[k], alpha_pr});
            }
            out.F_int[k] = gr.P_pr_plus[k] * alpha_pr_plus + gr.z_pr_plus[k];
        }
        return out;
    }
};

class MultibodySystem {
public:
    std::vector<SOABody*> bodies;
    std::vector<Joint*> joints;
    std::vector<FlexProperties*> flexs;
    ATBIFlex ATBI;
    Vec S0;

    explicit MultibodySystem(const std::vector<SOABody*>& bodies_) : bodies(bodies_), ATBI(bodies_) {
        std::vector<Vec> Theta0, Beta0, Eta0, EtaDot0;
        for (auto* b : bodies) {
            joints.push_back(&b->joint);
            flexs.push_back(&b->flex);
            Theta0.push_back(b->initialcondition.theta0);
            Beta0.push_back(b->initialcondition.beta0);
            Eta0.push_back(b->initialcondition.eta0);
            EtaDot0.push_back(b->initialcondition.eta_dot0);
        }
        S0 = SystemState(Theta0, Beta0, Eta0, EtaDot0).pack();
        ATBI.g << 0,0,0,0,0,9.81;
    }

    void setGravity(bool gravOnOff) {
        ATBI.g = Vec6::Zero();
        if (gravOnOff) ATBI.g << 0,0,0,0,0,9.81;
    }

    Vec EOM(double t, const Vec& S) const {
        SystemState state = SystemState::unpack(S, joints, flexs);
        for (int k = 0; k < static_cast<int>(bodies.size()); ++k) {
            if (bodies[k]->joint.type == "spherical" && state.Theta[k].size() >= 4) {
                double nrm = state.Theta[k].head(4).norm();
                if (nrm > 0) state.Theta[k].head(4) /= nrm;
            }
        }
        auto kin = ATBI.scatterKinematics(state);
        auto gr = ATBI.gatherATBI(state, kin, t);
        auto sc = ATBI.scatterATBI(kin, gr);
        std::vector<Vec> theta_dot, eta_dot_list;
        for (int k = 0; k < static_cast<int>(bodies.size()); ++k) {
            theta_dot.push_back(bodies[k]->joint.thetaDot(state.Theta[k], state.Beta[k]));
            eta_dot_list.push_back(state.Eta_dot[k]);
        }
        std::vector<Vec> blocks;
        blocks.insert(blocks.end(), theta_dot.begin(), theta_dot.end());
        blocks.insert(blocks.end(), sc.theta_ddot.begin(), sc.theta_ddot.end());
        blocks.insert(blocks.end(), eta_dot_list.begin(), eta_dot_list.end());
        blocks.insert(blocks.end(), sc.eta_ddot.begin(), sc.eta_ddot.end());
        return vstackVec(blocks);
    }
};

class Simulation {
public:
    struct Data {
        std::vector<double> time;
        std::vector<Vec> state_vector;
        std::vector<SystemState> state;
        std::vector<std::vector<Vec>> X_list, V_fl_list, a_fl_list, b_fl_list, alpha_fl_list;
        std::vector<std::vector<std::vector<Vec3>>> pos, pos_dot;
        std::vector<std::vector<Mat3>> R_i_list;
    } data;
    struct Setting {
        std::string solver{"RK4"};
        double atol{1e-3};
        double rtol{1e-6};
        double max_step{std::numeric_limits<double>::infinity()};
        double ani_dt{0.0};
    } setting;

    MultibodySystem& system;
    double tf{};
    double dt{};

    Simulation(MultibodySystem& system_, double tf_, double dt_) : system(system_), tf(tf_), dt(dt_) { setting.ani_dt = dt; }

    void integrateSystem(const std::string& solver = "RK4") {
        setting.solver = solver;
        std::vector<double> times;
        std::vector<Vec> states;
        if (solver == "RK4") integrateRK4(times, states);
        else if (solver == "BE") integrateBackwardEuler(times, states);
        else throw std::runtime_error("Only RK4 and BE are implemented in this self-contained C++ port. Use Boost.Odeint if you want adaptive methods.");

        double dt0 = setting.ani_dt;
        if (dt0 < dt) throw std::runtime_error("Invalid output timestep: ani_dt < sim dt");
        int scale = std::max(1, static_cast<int>(std::round(dt0 / dt)));
        int nt = static_cast<int>(states.size() / scale);
        data = Data{};
        for (int i = 0; i < nt; ++i) {
            int j = i * scale;
            double t = times[j];
            SystemState current_state = SystemState::unpack(states[j], system.joints, system.flexs);
            auto kin = system.ATBI.scatterKinematics(current_state);
            auto gr = system.ATBI.gatherATBI(current_state, kin, t);
            auto sc = system.ATBI.scatterATBI(kin, gr);
            data.time.push_back(t);
            data.state_vector.push_back(states[j]);
            data.state.push_back(std::move(current_state));
            data.X_list.push_back(kin.X);
            data.V_fl_list.push_back(kin.V);
            data.a_fl_list.push_back(kin.a_fl);
            data.b_fl_list.push_back(kin.b_fl);
            data.alpha_fl_list.push_back(sc.alpha_fl);
            data.pos.push_back(kin.pos);
            data.pos_dot.push_back(kin.pos_dot);
            data.R_i_list.push_back(kin.R_i);
        }
    }

    void setAniDt(double x) { setting.ani_dt = x; }
    void setTol(double atol, double rtol) { setting.atol = atol; setting.rtol = rtol; }
    void setMaxStep(double max_step) { setting.max_step = max_step; }

    // One wide CSV: t, state variables, and node positions. Easy to plot with pandas.
    void writeCSV(const std::string& filename) const {
        if (data.time.empty()) throw std::runtime_error("No simulation data. Call integrateSystem() first.");
        std::ofstream out(filename);
        if (!out) throw std::runtime_error("Could not open CSV file: " + filename);
        out << std::setprecision(17);
        out << "t";
        int stateSize = static_cast<int>(data.state_vector.front().size());
        for (int i = 0; i < stateSize; ++i) out << ",state_" << i;
        int nBodies = static_cast<int>(system.bodies.size());
        for (int b = 0; b < nBodies; ++b) {
            int nNodes = system.bodies[b]->flex.n_nd;
            for (int node = 0; node < nNodes; ++node) {
                out << ",body" << b << "_node" << node << "_x";
                out << ",body" << b << "_node" << node << "_y";
                out << ",body" << b << "_node" << node << "_z";
                out << ",body" << b << "_node" << node << "_vx";
                out << ",body" << b << "_node" << node << "_vy";
                out << ",body" << b << "_node" << node << "_vz";
            }
        }
        out << "\n";
        for (size_t i = 0; i < data.time.size(); ++i) {
            out << data.time[i];
            for (int s = 0; s < data.state_vector[i].size(); ++s) out << ',' << data.state_vector[i](s);
            for (int b = 0; b < nBodies; ++b) {
                for (int node = 0; node < system.bodies[b]->flex.n_nd; ++node) {
                    const Vec3& p = data.pos[i][b][node];
                    const Vec3& v = data.pos_dot[i][b][node];
                    out << ',' << p(0) << ',' << p(1) << ',' << p(2)
                        << ',' << v(0) << ',' << v(1) << ',' << v(2);
                }
            }
            out << "\n";
        }
    }

    // Long/tidy CSV useful for animation in Python: t, frame, body, node, x, y, z, vx, vy, vz.
    void writeNodeCSV(const std::string& filename) const {
        if (data.time.empty()) throw std::runtime_error("No simulation data. Call integrateSystem() first.");
        std::ofstream out(filename);
        if (!out) throw std::runtime_error("Could not open CSV file: " + filename);
        out << std::setprecision(17);
        out << "t,frame,body,node,x,y,z,vx,vy,vz\n";
        for (size_t i = 0; i < data.time.size(); ++i) {
            for (int b = 0; b < static_cast<int>(system.bodies.size()); ++b) {
                for (int node = 0; node < system.bodies[b]->flex.n_nd; ++node) {
                    const Vec3& p = data.pos[i][b][node];
                    const Vec3& v = data.pos_dot[i][b][node];
                    out << data.time[i] << ',' << i << ',' << b << ',' << node << ','
                        << p(0) << ',' << p(1) << ',' << p(2) << ','
                        << v(0) << ',' << v(1) << ',' << v(2) << "\n";
                }
            }
        }
    }

private:
    void integrateRK4(std::vector<double>& times, std::vector<Vec>& states) const {
        int steps = static_cast<int>(std::round(tf / dt));
        times.resize(steps + 1);
        states.resize(steps + 1);
        times[0] = 0.0;
        states[0] = system.S0;
        for (int i = 0; i < steps; ++i) {
            double t = times[i];
            const Vec& y = states[i];
            Vec k1 = system.EOM(t, y);
            Vec k2 = system.EOM(t + dt/2.0, y + (dt/2.0)*k1);
            Vec k3 = system.EOM(t + dt/2.0, y + (dt/2.0)*k2);
            Vec k4 = system.EOM(t + dt, y + dt*k3);
            states[i+1] = y + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4);
            times[i+1] = t + dt;
        }
    }

    void integrateBackwardEuler(std::vector<double>& times, std::vector<Vec>& states) const {
        int steps = static_cast<int>(std::round(tf / dt));
        times.resize(steps + 1);
        states.resize(steps + 1);
        times[0] = 0.0;
        states[0] = system.S0;
        const double tol = 1e-8;
        const int max_iter = 20;
        for (int i = 0; i < steps; ++i) {
            double t_next = times[i] + dt;
            Vec y_old = states[i];
            Vec y = y_old;
            for (int it = 0; it < max_iter; ++it) {
                Vec y_next = y_old + dt * system.EOM(t_next, y);
                if ((y_next - y).norm() < tol) { y = y_next; break; }
                y = y_next;
            }
            states[i+1] = y;
            times[i+1] = t_next;
        }
    }
};

} // namespace soa
