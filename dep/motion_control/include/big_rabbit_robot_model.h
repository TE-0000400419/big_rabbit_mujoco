#pragma once

// generator-stamp sha256: 017477ac2ce1255d

// tools/export_robot_model_header.py が生成。**このファイルを直接編集しない**。
//   数値の出どころ: robotmodel/big_rabbit/big_rabbit.xml（Isaac の USD と同じ MJCF。sim と実機で同じモデルを使うため）
//   ロジックの原本: tools/templates/big_rabbit_robot_model.h.in（ここを直して再生成する）
//
// 用途: 実機での接地判定と骨盤高の推定に使う運動学モデル。
// 構造体は RobotCalc（RCL_RobotModel.h）の link / robot をそのまま踏襲する。
// ただしリンク間に固定回転があるので link に R_fixed を足してある。
// RobotCalc の RNEA / 動力学を流用するときは、回転行列の更新を
//     r->l[i].R = r->l[i].R_fixed * rot33(r->l[i].a_joint, q(i));
// に変える（元は rot33 だけで上書きしている）。BigRabbitRobotUpdate はこの形で回す。
//
// 関節角の定義は MJCF / Isaac / policy と同一。bridge の joint_position がそのまま入る。
// モータ次元の符号・減速比は big_rabbit_model_param.h の担当でここには出てこない。
//
// 脚は左右独立の 5 関節直列鎖（base = 骨盤）。浮遊ベースの扱いは呼ぶ側。
//
// リンク構成:
//   left : pelvis -> left_hip_yaw -> left_hip_roll -> left_thigh -> left_shin -> left_foot
//   right: pelvis -> right_hip_yaw -> right_hip_roll -> right_thigh -> right_shin -> right_foot
// 全質量（骨盤 + 両脚）: 6.970040 kg

#include <Eigen/Dense>

#define BIG_RABBIT_LEG_JOINT_NUM (5)
#define BIG_RABBIT_LEG_NUM (2)

#ifndef JOINT_NUM
#define JOINT_NUM BIG_RABBIT_LEG_JOINT_NUM // 片脚の関節数
#define LINK_NUM (JOINT_NUM + 1)           // base + 関節数
#endif

using namespace Eigen;
using Vector6f = Eigen::Matrix<float, 6, 1>; // 作業空間ベクトル
using Matrix6f = Eigen::Matrix<float, 6, 6>;

using VectorJointSpacef = Eigen::Matrix<float, JOINT_NUM, 1>;         // 関節空間ベクトル
using VectorLinkSpacef = Eigen::Matrix<float, LINK_NUM, 1>;           // リンク空間ベクトル
using MatrixJointSpacef = Eigen::Matrix<float, JOINT_NUM, JOINT_NUM>; // 関節空間行列
using MatrixJacobif = Eigen::Matrix<float, 6, JOINT_NUM>;             // ヤコビ行列
using MatrixJacobiInvf = Eigen::Matrix<float, JOINT_NUM, 6>;          // ヤコビ行列の逆行列

enum axis
{
    x,
    y,
    z,
    minus_x,
    minus_y,
    minus_z,
    none
};

enum leg_side
{
    LEG_LEFT = 0,
    LEG_RIGHT = 1,
};

// NOTE: typedef にしない。POSIX の link()（unistd.h）と名前が衝突して
//       'redeclared as different kind of entity' になる。C++ では struct 宣言だけで
//       型名として使えるので、衝突する TU では struct link / rc_link と書く。
struct link
{
    int id;            // リンクID　base=0
    Vector3f p_child;  // iリンクローカル座標における、子リンクの位置ベクトル
                       // 先端リンクのときは、外力作用点（足裏中心）の位置ベクトル
    enum axis a_joint; // iリンクローカル座標における、ジョイントの回転方向
    Vector3f e_joint;  // iリンクローカル座標における、iジョイントの回転軸単位ベクトル
    Matrix3f I;        // iリンクローカル座標における、重心周りの慣性テンソル
    Vector3f s_com;    // iリンクローカル座標における、重心位置
    float m;           // iリンクの質量
    Matrix3f R_fixed;  // 親リンクからiリンクへの**固定**回転（関節角に依らない）
    Matrix3f R;        // 親リンクからiリンクへの回転行列 = R_fixed * rot33(a_joint, q)

    Vector3f a;     // iリンクローカル座標基準加速度
    Vector3f a_com; // iリンクローカル座標基準重心加速度
    Vector3f f;     // iリンクローカル座標基準　ローカル座標原点に作用する反力
    Vector3f f_com; // iリンクローカル座標基準　重心に作用する力(慣性力)
    Vector3f t;     // iリンクローカル座標基準　ローカル座標原点に作用する反モーメント
    Vector3f t_com; // iリンクローカル座標基準　重心に作用するモーメント(慣性力)
    float t_joint;  // 関節駆動トルク（反力の反対方向）
    Vector3f w;     // iリンクローカル座標基準 角速度
    Vector3f dw;    // iリンクローカル座標基準 角加速度
};

struct eq
{
    // 運動方程式
    MatrixJointSpacef Mq;  // 関節空間慣性行列
    VectorJointSpacef Cq;  // 関節空間コリオリ・遠心力ベクトル
    VectorJointSpacef Gq;  // 関節空間重力ベクトル
    VectorJointSpacef tau; // 関節駆動トルクベクトル
    VectorJointSpacef Fq;  // 一般化外力ベクトル
};

struct robot
{
    struct link base;
    struct link l[JOINT_NUM];
    Matrix3f R_tip;        // 基準座標系から先端座標系への回転行列
    Vector3f eulerZYX_tip; // 基準座標系から先端座標系へのオイラー角(ZYX)
    Vector4f quat_tip;     // 基準座標系から先端座標系へのクォータニオン(wxyz)
    Vector3f p_tip;        // 基準座標系から先端座標系への位置ベクトル

    MatrixJacobif J_tip;          // 先端ヤコビ行列
    MatrixJacobiInvf J_tip_srinv; // 先端ヤコビ行列の逆行列(SRInverse)

    struct eq dynamic_equation; // 動力学方程式
};

/// 足裏の接地カプセル（足リンク座標）。接地判定と骨盤高の逆算に使う。
struct foot_sole
{
    Vector3f p_a; // カプセル軸の端点 A
    Vector3f p_b; // カプセル軸の端点 B
    float radius; // カプセル半径。接地面は軸から radius だけ下
};

// link() が見えている TU 用の別名。struct link と書きたくない場所で使う。
using rc_link = struct link;
using rc_robot = struct robot;
using rc_foot_sole = struct foot_sole;

// ---- 以下、MJCF から落とした数値 ----

namespace big_rabbit_model_data
{
    struct LinkData
    {
        float p_child[3];  // 自リンク座標での子リンク原点（先端は足裏中心）
        float R_fixed[9];  // 親 <- 自リンク の固定回転（行優先）
        float e_joint[3];  // 自リンク座標での関節軸
        int a_joint;       // enum axis
        float s_com[3];    // 自リンク座標での重心
        float m;           // 質量 [kg]
        float I[9];        // 自リンク座標・重心まわりの慣性テンソル（行優先）
    };

    inline constexpr float kTotalMass = +6.97004f;

    // ---- left leg: base = pelvis ----
    inline constexpr LinkData kLeftBase = {
        {+0.0f, +0.06f, +0.00745f},  // p_child: left_hip_yaw の原点（骨盤座標）
        {+1.0f, +0.0f, +0.0f, +0.0f, +1.0f, +0.0f, +0.0f, +0.0f, +1.0f},  // R_fixed: base は単位行列
        {+0.0f, +0.0f, +0.0f}, none,  // base に関節は無い
        {+3.70396185e-20f, -0.000166811612f, +0.0296871098f},
        +1.20735f,
        {+0.00468713107f, +1.65585067e-06f, -1.28472037e-06f, +1.65585067e-06f, +0.000859551684f, +1.55112331e-06f, -1.28472037e-06f, +1.55112331e-06f, +0.00529399977f},
    };

    inline constexpr LinkData kLeftLink[BIG_RABBIT_LEG_JOINT_NUM] = {
        { // l[0] left_hip_yaw  (left_hip_yaw_joint, 可動 -120..+120 deg)
            {+9.76996262e-18f, -0.044f, +0.10375f},  // p_child: left_hip_roll の原点
            {-1.87469973e-33f, +1.0f, +0.0f, +1.0f, -1.87469973e-33f, -1.2246468e-16f, -1.2246468e-16f, +0.0f, -1.0f},
            {+0.0f, +0.0f, -1.0f}, minus_z,
            {-1.37903873e-18f, -0.0180180851f, +0.100478511f},
            +0.68244f,
            {+0.000572508225f, +1.21581057e-15f, +3.99999997e-08f, +1.21581058e-15f, +0.000734048057f, -6.02069454e-05f, +3.99999997e-08f, -6.02069454e-05f, +0.000474516978f},
        },
        { // l[1] left_hip_roll  (left_hip_roll_joint, 可動 -120..+120 deg)
            {+0.0f, -0.1035f, +0.05055f},  // p_child: left_thigh の原点
            {+0.0f, -1.0f, +0.0f, +0.0f, +0.0f, +1.0f, -1.0f, +0.0f, +0.0f},
            {+0.0f, +0.0f, +1.0f}, z,
            {+0.0f, -0.112760549f, +0.0506755281f},
            +0.68627f,
            {+0.000768708089f, +7.99999896e-08f, -9.9999506e-08f, +7.99999896e-08f, +0.000599137781f, -4.39734747e-06f, -9.9999506e-08f, -4.39734747e-06f, +0.000776981808f},
        },
        { // l[2] left_thigh  (left_hip_pitch_joint, 可動 -120..+120 deg)
            {-0.143f, +3.17523785e-17f, -0.04405f},  // p_child: left_shin の原点
            {+1.0f, -0.0f, +0.0f, +0.0f, +0.0f, +1.0f, -0.0f, -1.0f, +0.0f},
            {+0.0f, +0.0f, -1.0f}, minus_z,
            {-0.133375983f, +2.43637657e-17f, -0.0210716707f},
            +0.6991f,
            {+0.000520949066f, +3.99999996e-08f, +0.000188241157f, +3.99999996e-08f, +0.00127817954f, +2.97819245e-16f, +0.000188241157f, +2.97819241e-16f, +0.00143732586f},
        },
        { // l[3] left_shin  (left_knee_joint, 可動 -180..+120 deg)
            {-0.16842f, +0.00136f, +0.0522f},  // p_child: left_foot の原点
            {+1.0f, +1.11022302e-16f, -1.23259516e-32f, -1.11022302e-16f, +1.0f, -2.22044605e-16f, -1.23259516e-32f, +2.22044605e-16f, +1.0f},
            {+0.0f, +0.0f, -1.0f}, minus_z,
            {-0.15851995f, +0.000852238576f, +0.0714584881f},
            +0.5966f,
            {+0.000450400031f, +3.49685504e-05f, +0.000205076817f, +3.49685504e-05f, +0.00104905796f, -7.12659527e-06f, +0.000205076817f, -7.12659527e-06f, +0.00125157475f},
        },
        { // l[4] left_foot  (left_ankle_joint, 可動 -120..+120 deg)
            {+0.076827818f, -0.0449859575f, +0.00675f},  // p_child: 足裏中心（外力作用点）
            {-0.981627183f, +0.190808995f, +2.05060831e-16f, +0.190808995f, +0.981627183f, -1.87048799e-16f, -2.3698388e-16f, -1.44484734e-16f, -1.0f},
            {+0.0f, +0.0f, +1.0f}, z,
            {+0.0447769122f, -0.0349866408f, +0.04017f},
            +0.21775f,
            {+0.000571282804f, -0.000121350311f, -3.3689638e-05f, -0.000121350311f, +0.000219593176f, +2.30767192e-05f, -3.3689638e-05f, +2.30767192e-05f, +0.00067714978f},
        },
    };

    // left 足裏カプセル（left_foot 座標）
    inline constexpr float kLeftSoleA[3] = {+0.0891349982f, +0.0183289958f, +0.03475f};
    inline constexpr float kLeftSoleB[3] = {+0.0645206378f, -0.108300911f, +0.03475f};
    inline constexpr float kLeftSoleRadius = +0.028f;

    // ---- right leg: base = pelvis ----
    inline constexpr LinkData kRightBase = {
        {+0.0f, -0.06f, +0.00745f},  // p_child: right_hip_yaw の原点（骨盤座標）
        {+1.0f, +0.0f, +0.0f, +0.0f, +1.0f, +0.0f, +0.0f, +0.0f, +1.0f},  // R_fixed: base は単位行列
        {+0.0f, +0.0f, +0.0f}, none,  // base に関節は無い
        {+3.70396185e-20f, -0.000166811612f, +0.0296871098f},
        +1.20735f,
        {+0.00468713107f, +1.65585067e-06f, -1.28472037e-06f, +1.65585067e-06f, +0.000859551684f, +1.55112331e-06f, -1.28472037e-06f, +1.55112331e-06f, +0.00529399977f},
    };

    inline constexpr LinkData kRightLink[BIG_RABBIT_LEG_JOINT_NUM] = {
        { // l[0] right_hip_yaw  (right_hip_yaw_joint, 可動 -120..+120 deg)
            {+9.76996262e-18f, -0.044f, +0.10375f},  // p_child: right_hip_roll の原点
            {-1.87469973e-33f, +1.0f, +0.0f, +1.0f, -1.87469973e-33f, -1.2246468e-16f, -1.2246468e-16f, +0.0f, -1.0f},
            {+0.0f, +0.0f, -1.0f}, minus_z,
            {-1.37903873e-18f, -0.0180180851f, +0.100478511f},
            +0.68244f,
            {+0.000572508225f, +1.21581057e-15f, +3.99999997e-08f, +1.21581058e-15f, +0.000734048057f, -6.02069454e-05f, +3.99999997e-08f, -6.02069454e-05f, +0.000474516978f},
        },
        { // l[1] right_hip_roll  (right_hip_roll_joint, 可動 -120..+120 deg)
            {+1.26750944e-17f, +0.1035f, +0.05055f},  // p_child: right_thigh の原点
            {+0.0f, -1.0f, +0.0f, +0.0f, +0.0f, +1.0f, -1.0f, +0.0f, +0.0f},
            {+0.0f, +0.0f, +1.0f}, z,
            {+1.56143104e-17f, +0.112760549f, +0.0506755281f},
            +0.68627f,
            {+0.000768708089f, +7.99999896e-08f, +9.9999506e-08f, +7.99999896e-08f, +0.000599137781f, +4.39734747e-06f, +9.9999506e-08f, +4.39734747e-06f, +0.000776981808f},
        },
        { // l[2] right_thigh  (right_hip_pitch_joint, 可動 -120..+120 deg)
            {-0.143f, +3.17523785e-17f, -0.04405f},  // p_child: right_shin の原点
            {+1.0f, +0.0f, -2.4492936e-16f, -2.4492936e-16f, +0.0f, -1.0f, +0.0f, +1.0f, +0.0f},
            {+0.0f, +0.0f, +1.0f}, z,
            {-0.133375983f, +2.43637657e-17f, -0.0210716707f},
            +0.6991f,
            {+0.000520949066f, +3.99999996e-08f, +0.000188241157f, +3.99999996e-08f, +0.00127817954f, +2.97819245e-16f, +0.000188241157f, +2.97819241e-16f, +0.00143732586f},
        },
        { // l[3] right_shin  (right_knee_joint, 可動 -180..+120 deg)
            {-0.16842f, -0.00136f, +0.0522f},  // p_child: right_foot の原点
            {+1.0f, +1.11022302e-16f, -1.23259516e-32f, -1.11022302e-16f, +1.0f, -2.22044605e-16f, -1.23259516e-32f, +2.22044605e-16f, +1.0f},
            {+0.0f, +0.0f, +1.0f}, z,
            {-0.158608329f, -0.000872346226f, +0.0715717221f},
            +0.59497f,
            {+0.000446959238f, -3.54028908e-05f, +0.000201576984f, -3.54028908e-05f, +0.00104371744f, +5.49460234e-06f, +0.000201576984f, +5.49460234e-06f, +0.00124859885f},
        },
        { // l[4] right_foot  (right_ankle_joint, 可動 -120..+120 deg)
            {+0.076827818f, +0.0449859575f, +0.00675f},  // p_child: 足裏中心（外力作用点）
            {-0.981627183f, -0.190808995f, +2.05060831e-16f, -0.190808995f, +0.981627183f, +1.87048799e-16f, -2.3698388e-16f, +1.44484734e-16f, -1.0f},
            {+0.0f, +0.0f, -1.0f}, minus_z,
            {+0.0447769122f, +0.0349866408f, +0.04017f},
            +0.21775f,
            {+0.000571282154f, +0.000121350311f, -3.36889691e-05f, +0.000121350311f, +0.000219592526f, -2.30759678e-05f, -3.36889691e-05f, -2.30759678e-05f, +0.00067714978f},
        },
    };

    // right 足裏カプセル（right_foot 座標）
    inline constexpr float kRightSoleA[3] = {+0.0891349982f, -0.0183289958f, +0.03475f};
    inline constexpr float kRightSoleB[3] = {+0.0645206378f, +0.108300911f, +0.03475f};
    inline constexpr float kRightSoleRadius = +0.028f;

} // namespace big_rabbit_model_data

// ============================================================================
//  ここから下はモデルの使い方（tools/templates/big_rabbit_robot_model.h.in）
// ============================================================================

/// bridge の 10 関節配列（isaac_policy::kJointNames の並び）での index。
inline int big_rabbit_leg_joint_index(enum leg_side side, int leg_joint)
{
    return (side == LEG_LEFT ? 0 : BIG_RABBIT_LEG_JOINT_NUM) + leg_joint;
}

/// 関節回転（自リンク -> 親リンク）。RobotCalc の rot33 と同じ規約。
/// RCL_Dynamics.h / RNEA3.h の rot33 と衝突しないよう名前を変えてある。
inline Matrix3f big_rabbit_rot33(enum axis a, float angle)
{
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    Matrix3f R = Matrix3f::Identity();
    switch (a)
    {
    case x:       R << 1, 0, 0,  0, c, -s,  0, s, c; break;
    case y:       R << c, 0, s,  0, 1, 0,  -s, 0, c; break;
    case z:       R << c, -s, 0,  s, c, 0,  0, 0, 1; break;
    case minus_x: R << 1, 0, 0,  0, c, s,  0, -s, c; break;
    case minus_y: R << c, 0, -s,  0, 1, 0,  s, 0, c; break;
    case minus_z: R << c, s, 0,  -s, c, 0,  0, 0, 1; break;
    default: break;
    }
    return R;
}

inline void set_big_rabbit_link(struct link *dst, const big_rabbit_model_data::LinkData &src, int id)
{
    dst->id = id;
    dst->p_child = Vector3f(src.p_child[0], src.p_child[1], src.p_child[2]);
    dst->e_joint = Vector3f(src.e_joint[0], src.e_joint[1], src.e_joint[2]);
    dst->a_joint = static_cast<enum axis>(src.a_joint);
    dst->s_com = Vector3f(src.s_com[0], src.s_com[1], src.s_com[2]);
    dst->m = src.m;
    for (int row = 0; row < 3; row++)
    {
        for (int col = 0; col < 3; col++)
        {
            dst->R_fixed(row, col) = src.R_fixed[row * 3 + col];
            dst->I(row, col) = src.I[row * 3 + col];
        }
    }
    dst->R = dst->R_fixed; // q=0 の姿勢。BigRabbitRobotUpdate が毎周期更新する
    dst->w = Vector3f::Zero();
    dst->dw = Vector3f::Zero();
    dst->a = Vector3f::Zero();
    dst->t_joint = 0.0f;
}

/// 片脚（base = 骨盤）を組む。左右で別インスタンスを持つ。
inline void set_big_rabbit_leg(struct robot *r, enum leg_side side)
{
    using namespace big_rabbit_model_data;
    const LinkData &base = (side == LEG_LEFT) ? kLeftBase : kRightBase;
    const LinkData *links = (side == LEG_LEFT) ? kLeftLink : kRightLink;

    set_big_rabbit_link(&r->base, base, 0);
    r->base.R = Matrix3f::Identity();
    r->base.R_fixed = Matrix3f::Identity();
    // ベースの加速度は、マイナス重力加速度（骨盤座標。傾いているなら呼ぶ側で回す）
    r->base.a = Vector3f(0.0f, 0.0f, 9.81f);
    for (int i = 0; i < BIG_RABBIT_LEG_JOINT_NUM; i++)
    {
        set_big_rabbit_link(&r->l[i], links[i], i + 1);
    }
    r->R_tip = Matrix3f::Identity();
    r->p_tip = Vector3f::Zero();
    r->J_tip = MatrixJacobif::Zero();
}

/// 足裏カプセル（足リンク座標）を取り出す。
inline struct foot_sole big_rabbit_foot_sole(enum leg_side side)
{
    using namespace big_rabbit_model_data;
    struct foot_sole sole;
    const float *a = (side == LEG_LEFT) ? kLeftSoleA : kRightSoleA;
    const float *b = (side == LEG_LEFT) ? kLeftSoleB : kRightSoleB;
    sole.p_a = Vector3f(a[0], a[1], a[2]);
    sole.p_b = Vector3f(b[0], b[1], b[2]);
    sole.radius = (side == LEG_LEFT) ? kLeftSoleRadius : kRightSoleRadius;
    return sole;
}

// ---------------------------------------------------------------------------
//  共有モデルと順運動学
//
//  使い方:
//      BigRabbitRobotUpdate(joint_position, imu_quat);   // 毎周期 1 回
//      const BigRabbitState &s = BigRabbitRobotState();  // どこからでも読む
//
//  座標系:
//      *_base ... 骨盤固定座標（骨盤原点・骨盤姿勢）。関節角だけで決まる。
//      *_grav ... 骨盤原点・**重力基準**の姿勢（IMU の姿勢で回したもの）。
//                 骨盤高の逆算はこちら。IMU を渡さなければ base と同じ。
// ---------------------------------------------------------------------------

/// 片脚の状態。すべて「骨盤基準」。RobotCalc の robot 構造体が持たない量を足す。
struct BigRabbitLegState
{
    float q[BIG_RABBIT_LEG_JOINT_NUM];       // 関節角 [rad]（MJCF / policy と同じ定義）
    Matrix3f R_base[BIG_RABBIT_LEG_JOINT_NUM];  // 骨盤 <- 各リンク の回転
    Vector3f p_base[BIG_RABBIT_LEG_JOINT_NUM];  // 骨盤基準の各関節原点   pel_p_i
    Vector3f z_base[BIG_RABBIT_LEG_JOINT_NUM];  // 骨盤基準の各関節回転軸 pel_z_i
    Vector3f com_base[BIG_RABBIT_LEG_JOINT_NUM]; // 骨盤基準の各リンク重心

    Matrix3f R_foot_base; // 骨盤 <- 足リンク
    Vector3f p_foot_base; // 骨盤基準の足リンク原点
    Vector3f p_sole_base;   // 骨盤基準の足裏中心（接触フレーム）pel_p_E。**リンク固定点**
    Vector3f p_axis_a_base; // 足裏カプセル軸の端点（骨盤基準）。これもリンク固定点
    Vector3f p_axis_b_base;

    // 重力基準（骨盤原点）。骨盤高の逆算はこちらを見る。
    // *_contact_* は軸端点から重力方向へ半径ぶん下ろした点＝地面に当たる点。
    // 重力方向に依存するのでリンク固定ではない（ヤコビには使わない）。
    Vector3f p_sole_grav;
    Vector3f p_contact_a_grav;
    Vector3f p_contact_b_grav;

    // 足裏中心の幾何ヤコビ（骨盤基準）。上 3 行が並進、下 3 行が回転。
    //   Jv[:, i] = z_i x (p_E - p_i),  Jw[:, i] = z_i
    // 接触力の推定は F = pinv(J^T) tau。J^T F = tau の関係から。
    MatrixJacobif J_sole;

    Vector3f com_leg_base; // 脚だけの重心（骨盤基準）
    float mass_leg;        // 脚だけの質量
};

/// 共有モデル。1 個だけ持ち、毎周期 BigRabbitRobotUpdate で更新する。
struct BigRabbitState
{
    bool initialized;
    struct robot leg[BIG_RABBIT_LEG_NUM];      // RobotCalc の robot。RNEA へそのまま渡せる
    BigRabbitLegState state[BIG_RABBIT_LEG_NUM];

    Matrix3f R_grav_base;  // 重力基準 <- 骨盤（IMU 姿勢。未設定なら単位行列）
    Vector3f gravity_base; // 骨盤基準で見た重力方向の単位ベクトル（policy の projected gravity と同じ量）

    Vector3f com_base;     // 全身重心（骨盤基準、骨盤リンクも含む）
    Vector3f com_grav;     // 同じものを重力基準へ回したもの
    float mass_total;      // 全身質量
};

/// 共有インスタンス。inline 変数なので複数の TU から触っても 1 個。
inline BigRabbitState g_big_rabbit_state{};

inline BigRabbitState &BigRabbitRobotState()
{
    return g_big_rabbit_state;
}

inline BigRabbitLegState &BigRabbitLegStateOf(enum leg_side side)
{
    return g_big_rabbit_state.state[side];
}

/// モデルを組む。BigRabbitRobotUpdate が未初期化なら勝手に呼ぶので通常は不要。
inline void BigRabbitRobotSetup()
{
    BigRabbitState &s = g_big_rabbit_state;
    set_big_rabbit_leg(&s.leg[LEG_LEFT], LEG_LEFT);
    set_big_rabbit_leg(&s.leg[LEG_RIGHT], LEG_RIGHT);
    s.R_grav_base = Matrix3f::Identity();
    s.gravity_base = Vector3f(0.0f, 0.0f, -1.0f);
    s.mass_total = big_rabbit_model_data::kTotalMass;
    s.initialized = true;
}

/// IMU の姿勢 quaternion (w,x,y,z) を入れる。重力基準の量を使うときだけ必要。
inline void BigRabbitRobotSetBaseOrientation(const float quat_wxyz[4])
{
    BigRabbitState &s = g_big_rabbit_state;
    const Quaternionf q(quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]);
    s.R_grav_base = q.normalized().toRotationMatrix();
    s.gravity_base = s.R_grav_base.transpose() * Vector3f(0.0f, 0.0f, -1.0f);
}

/// 片脚の順運動学。q は関節 5 個ぶん。robot 構造体の R / p_tip / J_tip も更新する。
inline void BigRabbitLegForwardKinematics(enum leg_side side, const float *q)
{
    BigRabbitState &s = g_big_rabbit_state;
    struct robot &r = s.leg[side];
    BigRabbitLegState &out = s.state[side];

    Matrix3f R = Matrix3f::Identity();  // 骨盤 <- 現在のリンク
    Vector3f p = r.base.p_child;        // 骨盤基準の link0 原点

    out.com_leg_base = Vector3f::Zero();
    out.mass_leg = 0.0f;
    for (int i = 0; i < BIG_RABBIT_LEG_JOINT_NUM; i++)
    {
        // ここが RobotCalc との唯一の違い。固定回転を掛けてから関節回転を掛ける。
        r.l[i].R = r.l[i].R_fixed * big_rabbit_rot33(r.l[i].a_joint, q[i]);
        R = R * r.l[i].R;

        out.q[i] = q[i];
        out.R_base[i] = R;
        out.p_base[i] = p;
        out.z_base[i] = R * r.l[i].e_joint;
        out.com_base[i] = p + R * r.l[i].s_com;

        out.com_leg_base += r.l[i].m * out.com_base[i];
        out.mass_leg += r.l[i].m;

        if (i + 1 < BIG_RABBIT_LEG_JOINT_NUM)
        {
            p += R * r.l[i].p_child; // 次のリンク原点へ
        }
    }
    if (out.mass_leg > 0.0f)
    {
        out.com_leg_base /= out.mass_leg;
    }

    const int tip = BIG_RABBIT_LEG_JOINT_NUM - 1;
    out.R_foot_base = R;
    out.p_foot_base = p;
    out.p_sole_base = p + R * r.l[tip].p_child; // 先端リンクの p_child = 足裏中心

    const struct foot_sole sole = big_rabbit_foot_sole(side);
    out.p_axis_a_base = p + R * sole.p_a;
    out.p_axis_b_base = p + R * sole.p_b;

    // 重力基準へ回してから、半径ぶん真下へ下ろす。これが地面に当たる点。
    const Vector3f down_grav(0.0f, 0.0f, -1.0f);
    out.p_sole_grav = s.R_grav_base * out.p_sole_base;
    out.p_contact_a_grav = s.R_grav_base * out.p_axis_a_base + sole.radius * down_grav;
    out.p_contact_b_grav = s.R_grav_base * out.p_axis_b_base + sole.radius * down_grav;

    // 足裏中心の幾何ヤコビ（骨盤基準）
    for (int i = 0; i < BIG_RABBIT_LEG_JOINT_NUM; i++)
    {
        const Vector3f arm = out.p_sole_base - out.p_base[i];
        out.J_sole.block<3, 1>(0, i) = out.z_base[i].cross(arm);
        out.J_sole.block<3, 1>(3, i) = out.z_base[i];
    }

    r.R_tip = out.R_foot_base;
    r.p_tip = out.p_sole_base;
    r.J_tip = out.J_sole;
    const Quaternionf quat(out.R_foot_base);
    r.quat_tip = Vector4f(quat.w(), quat.x(), quat.y(), quat.z());
    r.eulerZYX_tip = out.R_foot_base.eulerAngles(2, 1, 0);
}

/// 毎周期これを呼ぶ。joint_rad は bridge の 10 関節配列（kJointNames の並び）。
/// imu_quat_wxyz は省略可（nullptr）。渡すと *_grav 系と gravity_base が使えるようになる。
inline void BigRabbitRobotUpdate(const float *joint_rad, const float *imu_quat_wxyz = nullptr)
{
    BigRabbitState &s = g_big_rabbit_state;
    if (!s.initialized)
    {
        BigRabbitRobotSetup();
    }
    if (imu_quat_wxyz != nullptr)
    {
        BigRabbitRobotSetBaseOrientation(imu_quat_wxyz);
    }
    BigRabbitLegForwardKinematics(LEG_LEFT, joint_rad + 0);
    BigRabbitLegForwardKinematics(LEG_RIGHT, joint_rad + BIG_RABBIT_LEG_JOINT_NUM);

    // 全身重心。骨盤リンク（両脚で共通）を 1 回だけ足す。
    const struct link &pelvis = s.leg[LEG_LEFT].base;
    Vector3f weighted = pelvis.m * pelvis.s_com;
    float mass = pelvis.m;
    for (int side = 0; side < BIG_RABBIT_LEG_NUM; side++)
    {
        weighted += s.state[side].mass_leg * s.state[side].com_leg_base;
        mass += s.state[side].mass_leg;
    }
    s.mass_total = mass;
    s.com_base = weighted / mass;
    s.com_grav = s.R_grav_base * s.com_base;
}
