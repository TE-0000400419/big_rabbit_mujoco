// Big Rabbit の MuJoCo 実行本体。
// 物理シミュレーション、制御 bridge、Isaac policy 用観測生成、描画入力をここで接続する。
//
// 周期は 3 系統。
//   物理     1 kHz
//   ドライバ 1 kHz   （物理 step ごと。motor_angle_ref を追う）
//   policy  62.5 Hz  （16 物理 step ごと。obs54 -> 推論 -> joint_ref）

#include <big_rabbit_control_bridge.h>
#include <big_rabbit_model_param.h>
#include <isaac_policy_config.h>
#include <motion_command_source.h>
#include <sim_main.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <thread>

namespace
{

  constexpr std::size_t kJointNum = BigRabbitControlBridge::kJointAxisNum;

  enum class BridgeControlState
  {
    StateCrouch,
    StateRLControl,
    /// policy を外し、参照姿勢そのまま（action=0）を流す。追従性能の測定用。
    StateReferenceOnly,
  };

  /// BIG_RABBIT_CONTROL_MODE の解釈。
  BridgeControlState ControlModeFromEnv()
  {
    const char *text = std::getenv("BIG_RABBIT_CONTROL_MODE");
    const std::string mode = text && text[0] != '\0' ? std::string(text) : "rl";
    if (mode == "home" || mode == "HOME" || mode == "off" || mode == "OFF")
    {
      return BridgeControlState::StateCrouch;
    }
    if (mode == "reference" || mode == "REFERENCE")
    {
      return BridgeControlState::StateReferenceOnly;
    }
    return BridgeControlState::StateRLControl;
  }

  struct BigRabbitSimBridgeState
  {
    // MuJoCo 配列の index を名前解決後に保持する。
    std::array<int, kJointNum> qpos_indices{};
    std::array<int, kJointNum> dof_indices{};
    std::array<int, kJointNum> actuator_indices{};
    // free joint root の qpos/qvel 先頭 index。
    int root_qpos_adr = -1;
    int root_dof_adr = -1;
    // policy 観測に使う骨盤 body と左右足 body の id。
    int pelvis_body_id = -1;
    std::array<int, 2> foot_body_ids{-1, -1};
    // policy step 数。歩容位相はこれから作る（data->time ではなく整数カウント）。
    long long policy_step_count = 0;

    // ---- 計測（段 6 の判定に使う）----
    // RL 開始時点を基準にした変位と、体幹座標系での速度・yaw 速度の平均。
    // Isaac 側の評価指標（body_vx_mean_mps / yaw_rate_mean_rad_s）と同じ量。
    bool measure_started = false;
    double start_x = 0.0;
    double start_y = 0.0;
    double start_time = 0.0;
    double sum_body_vx = 0.0;
    double sum_body_vy = 0.0;
    double sum_yaw_rate = 0.0;
    double sum_contact = 0.0;
    long long measure_count = 0;
    double min_pelvis_z = 1.0e9;
    double max_torque_ratio = 0.0;
  };

  BigRabbitControlBridge bridge;
  BigRabbitSimBridgeState g_state;

  /// GLFW を初期化していない headless でも使える実時間。
  double WallClock()
  {
    using clock = std::chrono::steady_clock;
    static const auto start = clock::now();
    return std::chrono::duration<double>(clock::now() - start).count();
  }

  double EnvDouble(const char *name, double default_value)
  {
    const char *text = std::getenv(name);
    if (!text || text[0] == '\0')
    {
      return default_value;
    }
    return std::atof(text);
  }

  void BuildActuatorMapping(mjModel *model)
  {
    // 関節名から MuJoCo の joint id / qpos index / dof index を引く。
    // 名前は isaac_policy::kJointNames（生成物）を使うので、学習側と順序がずれない。
    std::array<int, kJointNum> joint_ids{};
    g_state.actuator_indices.fill(-1);

    for (std::size_t index = 0; index < kJointNum; ++index)
    {
      const int joint_id = mj_name2id(model, mjOBJ_JOINT, isaac_policy::kJointNames[index]);
      if (joint_id < 0)
      {
        std::cerr << "[FATAL] joint not found: " << isaac_policy::kJointNames[index] << std::endl;
        std::exit(1);
      }
      joint_ids[index] = joint_id;
      g_state.qpos_indices[index] = model->jnt_qposadr[joint_id];
      g_state.dof_indices[index] = model->jnt_dofadr[joint_id];
    }

    // actuator がどの joint を駆動するかを見て、関節順の actuator index を作る。
    for (int actuator_id = 0; actuator_id < model->nu; ++actuator_id)
    {
      const int joint_id = model->actuator_trnid[2 * actuator_id];
      for (std::size_t index = 0; index < kJointNum; ++index)
      {
        if (joint_id == joint_ids[index])
        {
          g_state.actuator_indices[index] = actuator_id;
          break;
        }
      }
    }
    for (std::size_t index = 0; index < kJointNum; ++index)
    {
      if (g_state.actuator_indices[index] < 0)
      {
        std::cerr << "[FATAL] actuator not found for " << isaac_policy::kJointNames[index] << std::endl;
        std::exit(1);
      }
    }

    const int root_joint_id = mj_name2id(model, mjOBJ_JOINT, "root");
    g_state.root_qpos_adr = model->jnt_qposadr[root_joint_id];
    g_state.root_dof_adr = model->jnt_dofadr[root_joint_id];

    g_state.pelvis_body_id = mj_name2id(model, mjOBJ_BODY, "pelvis");
    g_state.foot_body_ids[0] = mj_name2id(model, mjOBJ_BODY, isaac_policy::kFootBodyNames[0]);
    g_state.foot_body_ids[1] = mj_name2id(model, mjOBJ_BODY, isaac_policy::kFootBodyNames[1]);

    std::cout << "[INIT] MuJoCo mapping ready. actuators=" << model->nu << std::endl;
  }

  /// 左右足の接触法線力 [N] を集計する。**閾値化はしない**（bridge の仕事）。
  /// 実機では足裏力センサがこの値を返す。
  std::array<float, 2> ComputeFootForces(const mjModel *model, const mjData *data)
  {
    std::array<float, 2> normal_forces{};
    for (int contact_index = 0; contact_index < data->ncon; ++contact_index)
    {
      const mjContact &contact = data->contact[contact_index];
      const int body1 = model->geom_bodyid[contact.geom1];
      const int body2 = model->geom_bodyid[contact.geom2];
      double contact_force[6]{};
      mj_contactForce(model, data, contact_index, contact_force);

      for (std::size_t foot_index = 0; foot_index < normal_forces.size(); ++foot_index)
      {
        const int foot_body_id = g_state.foot_body_ids[foot_index];
        if (body1 == foot_body_id || body2 == foot_body_id)
        {
          normal_forces[foot_index] += static_cast<float>(std::max(0.0, contact_force[0]));
        }
      }
    }

    return normal_forces;
  }

  void InitializeRobotModel(mjModel *model, mjData *data)
  {
    // free root を原点上の起動高さに置く。
    const std::array<double, 7> root_qpos_target = {
        0.0, 0.0, isaac_policy::kSpawnHeightM, 1.0, 0.0, 0.0, 0.0,
    };
    for (int index = 0; index < 7; ++index)
    {
      data->qpos[g_state.root_qpos_adr + index] = root_qpos_target[index];
    }
    for (int index = 0; index < 6; ++index)
    {
      data->qvel[g_state.root_dof_adr + index] = 0.0;
    }

    // 10 関節を crouch へ初期化する。
    for (std::size_t index = 0; index < kJointNum; ++index)
    {
      data->qpos[g_state.qpos_indices[index]] = isaac_policy::kCrouchJointPositionRad[index];
      data->qvel[g_state.dof_indices[index]] = 0.0;
    }
    for (int actuator_id = 0; actuator_id < model->nu; ++actuator_id)
    {
      data->ctrl[actuator_id] = 0.0;
    }
    mj_forward(model, data);
  }

  void ReadJointState(const mjData *data, std::array<float, kJointNum> &pos,
                      std::array<float, kJointNum> &vel, std::array<float, kJointNum> &torque)
  {
    for (std::size_t index = 0; index < kJointNum; ++index)
    {
      const int qpos_index = g_state.qpos_indices[index];
      const int dof_index = g_state.dof_indices[index];
      pos[index] = static_cast<float>(data->qpos[qpos_index]);
      vel[index] = static_cast<float>(data->qvel[dof_index]);
      torque[index] = static_cast<float>(data->qfrc_actuator[dof_index]);
    }
  }

  void ApplyJointTorqueCommands(mjModel *model, mjData *data,
                                const std::array<float, kJointNum> &joint_torque_commands)
  {
    // ドライバ出力（関節側トルク）を MuJoCo actuator の ctrl 値へ変換する。
    for (std::size_t index = 0; index < kJointNum; ++index)
    {
      const int actuator_id = g_state.actuator_indices[index];
      double gear = model->actuator_gear[actuator_id * 6];
      if (std::abs(gear) < 1.0e-12)
      {
        gear = 1.0;
      }
      double control_value = static_cast<double>(joint_torque_commands[index]) / gear;
      const double control_low = model->actuator_ctrlrange[actuator_id * 2];
      const double control_high = model->actuator_ctrlrange[actuator_id * 2 + 1];
      if (control_low < control_high)
      {
        control_value = std::clamp(control_value, control_low, control_high);
      }
      data->ctrl[actuator_id] = control_value;
    }
  }

  void InitializeControlBridge(mjModel *model, mjData *data)
  {
    BuildActuatorMapping(model);

    std::cout << "=== Control Bridge Initialization ===" << std::endl;
    bridge.Initialize();
    std::cout << "=== Control Bridge Setup ===" << std::endl;
    bridge.ExecuteReady();

    InitializeRobotModel(model, data);

    std::cout << "[INIT] Big Rabbit sim bridge ready." << std::endl;
    const auto mode = ControlModeFromEnv();
    std::cout << "[INIT] control_mode="
              << (mode == BridgeControlState::StateCrouch
                      ? "crouch"
                      : (mode == BridgeControlState::StateReferenceOnly ? "reference" : "rl"))
              << std::endl;

  }

}  // namespace

// MuJoCo/GLFW の callback から参照するのでファイルスコープに置く。
mjModel *m = nullptr;
mjData *d = nullptr;

mjvCamera cam;
mjvOption opt;
mjvScene scn;
mjrContext con;

int button_left = 0;
int button_middle = 0;
int button_right = 0;
double lastx = 0.0;
double lasty = 0.0;

int main()
{
#ifdef BIG_RABBIT_SCENE_XML_DEFAULT
  const char *default_filename = BIG_RABBIT_SCENE_XML_DEFAULT;
#else
  const char *default_filename = "robotmodel/big_rabbit/scene.xml";
#endif

  const char *filename = std::getenv("BIG_RABBIT_SCENE_XML");
  if (!filename || filename[0] == '\0')
  {
    filename = default_filename;
  }
  std::cout << "[Config] scene xml: " << filename << std::endl;

  char error[1000];
  m = mj_loadXML(filename, nullptr, error, sizeof(error));
  if (!m)
  {
    std::cerr << "[FATAL] mj_loadXML failed: " << error << std::endl;
    return 1;
  }

  // MJCF 側の timestep ではなく、Isaac と同じ 1 kHz を使う。
  m->opt.timestep = PHYSICS_TIMESTEP;
  const double max_sim_time = EnvDouble("BIG_RABBIT_MAX_SIM_TIME", 0.0);
  const int physics_steps_per_control =
      static_cast<int>(CONTROL_TIMESTEP / PHYSICS_TIMESTEP + 0.5);
  // 空文字を「設定済み」と誤解しないようにする。
  const char *headless_text = std::getenv("BIG_RABBIT_HEADLESS");
  const bool headless = headless_text != nullptr && headless_text[0] != '\0';
  // 描画せず実時間に同期するモード。ゲームパッド操作の確認に使う
  // （headless のまま一気に回すと sim 時刻が実時間より速く進み、watchdog が発火し続ける）。
  const char *realtime_text = std::getenv("BIG_RABBIT_REALTIME");
  const bool realtime = realtime_text != nullptr && realtime_text[0] != '\0';

  std::cout << "[Config] Physics timestep: " << PHYSICS_TIMESTEP << " s ("
            << (1.0 / PHYSICS_TIMESTEP) << " Hz)" << std::endl;
  std::cout << "[Config] Policy timestep : " << CONTROL_TIMESTEP << " s ("
            << (1.0 / CONTROL_TIMESTEP) << " Hz)" << std::endl;
  std::cout << "[Config] Driver timestep : " << DRIVER_TIMESTEP << " s ("
            << (1.0 / DRIVER_TIMESTEP) << " Hz)" << std::endl;
  std::cout << "[Config] Physics steps per policy step: " << physics_steps_per_control << std::endl;
  if (max_sim_time > 0.0)
  {
    std::cout << "[Config] max sim time: " << max_sim_time << " s" << std::endl;
  }
  std::cout << "[Config] headless: " << (headless ? "yes" : "no") << std::endl;

  d = mj_makeData(m);

  GLFWwindow *window = nullptr;
  if (!headless)
  {
    glfwInit();
    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "big_rabbit_mujoco_sim", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetScrollCallback(window, scroll_callback);

    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    cam.type = mjCAMERA_FREE;
    cam.lookat[0] = 0.0;
    cam.lookat[1] = 0.0;
    cam.lookat[2] = 0.30;
    cam.distance = 2.2;
    cam.azimuth = 150.0;
    cam.elevation = -15.0;
  }

  InitializeControlBridge(m, d);

  // 起動から何秒で RL に切り替えるか。既定 1 s は crouch で落ち着かせるため。
  const double rl_start_time = EnvDouble("BIG_RABBIT_RL_START_S", 1.0);
  const double log_interval = EnvDouble("BIG_RABBIT_LOG_INTERVAL_S", 1.0);
  const BridgeControlState requested_mode = ControlModeFromEnv();

  // 移動指令の入力元。env（固定）/ udp（ゲームパッド）/ sequence（指令列）。
  // ゲームパッドは別プロセス（tools/gamepad_command.py）が読み、UDP で送ってくる。
  // 分離しておくと、送信側が落ちても受信側の watchdog が指令をゼロへ落とせる。
  auto command_source = MakeMotionCommandSourceFromEnv();
  std::cout << "[Config] command source: " << command_source->name();
  if (!command_source->status().empty())
  {
    std::cout << "  (" << command_source->status() << ")";
  }
  std::cout << std::endl;

  // qpos 軌跡を CSV に吐く（動画描画用）。物理と制御は C++ 側が正本で、
  // Python 側は描画だけを担う。
  std::FILE *qpos_csv = nullptr;
  const double qpos_interval = 1.0 / EnvDouble("BIG_RABBIT_QPOS_HZ", 50.0);
  double next_qpos_time = 0.0;
  if (const char *qpos_path = std::getenv("BIG_RABBIT_QPOS_CSV"))
  {
    qpos_csv = std::fopen(qpos_path, "w");
    if (!qpos_csv)
    {
      std::cerr << "[FATAL] qpos CSV を開けない: " << qpos_path << std::endl;
      return 1;
    }
    std::fprintf(qpos_csv, "t");
    for (int i = 0; i < m->nq; ++i) std::fprintf(qpos_csv, ",q%d", i);
    std::fprintf(qpos_csv, "\n");
    std::cout << "[Config] qpos csv: " << qpos_path << " @ "
              << (1.0 / qpos_interval) << " Hz" << std::endl;
  }

  // 指令と実測を CSV に吐く（追従性能の解析用）。解析は Python 側で行う。
  std::FILE *trace = nullptr;
  if (const char *trace_path = std::getenv("BIG_RABBIT_TRACE_CSV"))
  {
    trace = std::fopen(trace_path, "w");
    if (!trace)
    {
      std::cerr << "[FATAL] trace CSV を開けない: " << trace_path << std::endl;
      return 1;
    }
    std::fprintf(trace, "t");
    for (std::size_t i = 0; i < kJointNum; ++i) std::fprintf(trace, ",cmd_%s", isaac_policy::kJointNames[i]);
    for (std::size_t i = 0; i < kJointNum; ++i) std::fprintf(trace, ",act_%s", isaac_policy::kJointNames[i]);
    std::fprintf(trace, ",tau_ratio");
    // 関節ごとのトルク使用率と接地を後ろに足す。既存列の位置は変えない。
    for (std::size_t i = 0; i < kJointNum; ++i) std::fprintf(trace, ",ratio_%s", isaac_policy::kJointNames[i]);
    for (std::size_t i = 0; i < kJointNum; ++i) std::fprintf(trace, ",tau_j_%s", isaac_policy::kJointNames[i]);
    std::fprintf(trace, ",contact_l,contact_r,phase\n");
    std::cout << "[Config] trace csv: " << trace_path << std::endl;
  }

  const double wall_t0 = WallClock();
  const double draw_dt = 1.0 / 30.0;
  double next_draw_time = 0.0;
  double next_control_time = rl_start_time;
  double next_log_time = 0.0;
  const double eps = 1.0e-12;
  bool rl_reset = true;

  while ((max_sim_time <= 0.0 || d->time + eps < max_sim_time) &&
         (headless || !glfwWindowShouldClose(window)))
  {
    // headless では実時間同期しない。描画時だけ wall clock に追従させる。
    // substep 上限は「描画が固まらないように」のためなので headless では掛けない
    // （掛けると 1000 step = 1 s で外側 loop を抜けて打ち切られてしまう）。
    const double wall_elapsed =
        (headless && !realtime) ? max_sim_time : (WallClock() - wall_t0);
    const int substep_limit = (headless && !realtime) ? std::numeric_limits<int>::max() : 1000;
    int substeps = 0;

    while (d->time + eps < wall_elapsed && substeps < substep_limit)
    {
      std::array<float, kJointNum> pos{};
      std::array<float, kJointNum> vel{};
      std::array<float, kJointNum> torque{};
      std::array<float, 4> imu_quat{1.0f, 0.0f, 0.0f, 0.0f};
      std::array<float, 3> imu_gyro{};
      std::array<float, 3> imu_accel{};

      ReadJointState(d, pos, vel, torque);
      for (int index = 0; index < 4; ++index)
      {
        imu_quat[index] = static_cast<float>(d->qpos[g_state.root_qpos_adr + 3 + index]);
      }
      for (int index = 0; index < 3; ++index)
      {
        imu_gyro[index] = static_cast<float>(d->qvel[g_state.root_dof_adr + 3 + index]);
        imu_accel[index] = static_cast<float>(d->qacc[g_state.root_dof_adr + index]);
      }

      // ---- センサ生値を bridge へ。変換（重力方向・接地判定）は bridge の中で行う ----
      bridge.SetSensorDataJoint(pos, vel, torque);
      bridge.SetSensorDataIMU(imu_quat, imu_gyro, imu_accel);
      bridge.SetSensorDataFootForce(ComputeFootForces(m, d));
      // 骨盤高は対応するセンサが無い。sim では真値、実機では支持脚から運動学で逆算する。
      bridge.SetSensorDataBaseHeight(static_cast<float>(d->xpos[3 * g_state.pelvis_body_id + 2]));

      // ---- policy 周期（62.5 Hz）----
      BridgeControlState control_state = BridgeControlState::StateCrouch;
      if (d->time + eps >= rl_start_time)
      {
        control_state = requested_mode;
      }

      if (d->time + eps >= next_control_time || control_state == BridgeControlState::StateCrouch)
      {
        if (control_state == BridgeControlState::StateRLControl)
        {
          // 指令は policy 周期で更新する。RL 開始からの経過時刻を渡す
          // （sequence の時刻基準と watchdog の時刻基準を揃えるため）。
          command_source->Poll(d->time - rl_start_time);
          bridge.SetMotionCommand(command_source->command());
          bridge.ExecuteRLControl(g_state.policy_step_count, rl_reset);
          rl_reset = false;
          g_state.policy_step_count++;
          next_control_time += CONTROL_TIMESTEP;
        }
        else if (control_state == BridgeControlState::StateReferenceOnly)
        {
          bridge.ExecuteReferenceOnly(g_state.policy_step_count);
          g_state.policy_step_count++;
          next_control_time += CONTROL_TIMESTEP;
        }
        else
        {
          // crouch 保持。policy step は進めない（位相は RL 開始時点を 0 にする）。
          bridge.ExecuteJointSpaceControl();
        }
      }

      // ---- 計測 ----
      if (control_state == BridgeControlState::StateRLControl)
      {
        const double *pelvis_pos = d->xpos + 3 * g_state.pelvis_body_id;
        if (!g_state.measure_started)
        {
          g_state.measure_started = true;
          g_state.start_x = pelvis_pos[0];
          g_state.start_y = pelvis_pos[1];
          g_state.start_time = d->time;
        }
        double pelvis_vel[6]{};
        mj_objectVelocity(m, d, mjOBJ_XBODY, g_state.pelvis_body_id, pelvis_vel, 1);
        // rot:lin の順。lin は body frame（flg_local=1）なので Isaac の root_lin_vel_b と同じ。
        g_state.sum_body_vx += pelvis_vel[3];
        g_state.sum_body_vy += pelvis_vel[4];
        g_state.sum_yaw_rate += pelvis_vel[2];
        const auto contacts = bridge.GetFeetContact();
        g_state.sum_contact += 0.5 * (contacts[0] + contacts[1]);
        g_state.measure_count++;
        g_state.min_pelvis_z = std::min(g_state.min_pelvis_z, pelvis_pos[2]);
        g_state.max_torque_ratio =
            std::max(g_state.max_torque_ratio,
                     static_cast<double>(bridge.GetIsaacPolicyDebugData().driver_torque_ratio));
      }

      // ---- ドライバ周期（1 kHz）----
      bridge.ExecuteDriverStep();
      std::array<float, kJointNum> ctrl_out{};
      bridge.GetControlCommandsJoint(ctrl_out);
      ApplyJointTorqueCommands(m, d, ctrl_out);

      // 追従性能の解析用に、指令と実測をそのまま吐く。
      if (trace && control_state != BridgeControlState::StateCrouch)
      {
        const auto debug = bridge.GetIsaacPolicyDebugData();
        std::fprintf(trace, "%.6f", d->time);
        for (std::size_t i = 0; i < kJointNum; ++i)
        {
          std::fprintf(trace, ",%.9g", debug.joint_position_ref[i]);
        }
        for (std::size_t i = 0; i < kJointNum; ++i)
        {
          std::fprintf(trace, ",%.9g", d->qpos[g_state.qpos_indices[i]]);
        }
        std::fprintf(trace, ",%.9g", debug.driver_torque_ratio);
        std::array<float, kJointNum> motor_torque{};
        std::array<float, kJointNum> joint_torque{};
        bridge.GetMotorTorqueRef(motor_torque);
        bridge.GetControlCommandsJoint(joint_torque);
        for (std::size_t i = 0; i < kJointNum; ++i)
        {
          std::fprintf(trace, ",%.9g", motor_torque[i] / big_rabbit::MotorEffortLimitNm(i));
        }
        for (std::size_t i = 0; i < kJointNum; ++i)
        {
          std::fprintf(trace, ",%.9g", joint_torque[i]);
        }
        const auto trace_contacts = bridge.GetFeetContact();
        std::fprintf(trace, ",%.0f,%.0f,%.6f\n", trace_contacts[0], trace_contacts[1], debug.phase_unit);
      }


      mj_step(m, d);
      ++substeps;

      if (qpos_csv && d->time + eps >= next_qpos_time)
      {
        std::fprintf(qpos_csv, "%.6f", d->time);
        for (int i = 0; i < m->nq; ++i) std::fprintf(qpos_csv, ",%.9g", d->qpos[i]);
        std::fprintf(qpos_csv, "\n");
        next_qpos_time += qpos_interval;
      }

      if (log_interval > 0.0 && d->time + eps >= next_log_time)
      {
        const auto debug = bridge.GetIsaacPolicyDebugData();
        const double *pelvis_pos = d->xpos + 3 * g_state.pelvis_body_id;
        std::cout << "[SIM] t=" << d->time << " pelvis_z=" << pelvis_pos[2]
                  << " upright=" << -bridge.GetProjectedGravity()[2] << " phase=" << debug.phase_unit
                  << " tau_ratio=" << debug.driver_torque_ratio << std::endl;
        next_log_time += log_interval;
      }
    }

    if (headless)
    {
      if (realtime)
      {
        // 実時間に追いついたら少し待つ。描画はしない。
        if (d->time > WallClock() - wall_t0)
        {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        continue;
      }
      break;
    }

    if (d->time + eps >= next_draw_time)
    {
      mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
      int width = 0;
      int height = 0;
      glfwGetFramebufferSize(window, &width, &height);
      const mjrRect viewport = {0, 0, width, height};
      mjr_render(viewport, &scn, &con);
      glfwSwapBuffers(window);
      next_draw_time += draw_dt;
    }

    glfwPollEvents();

    if (d->time > wall_elapsed)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  if (trace)
  {
    std::fclose(trace);
  }
  if (qpos_csv)
  {
    std::fclose(qpos_csv);
  }

  std::cout << "[DONE] sim time: " << d->time << " s" << std::endl;

  if (g_state.measure_count > 0)
  {
    const double n = static_cast<double>(g_state.measure_count);
    const double *pelvis_pos = d->xpos + 3 * g_state.pelvis_body_id;
    const double elapsed = d->time - g_state.start_time;
    const auto command = bridge.GetIsaacPolicyDebugData().motion_command;
    std::cout << "[EVAL] command       = (" << command[0] << ", " << command[1] << ", "
              << command[2] << ")" << std::endl;
    std::cout << "[EVAL] rl_time_s     = " << elapsed << std::endl;
    std::cout << "[EVAL] body_vx_mean  = " << g_state.sum_body_vx / n << " m/s" << std::endl;
    std::cout << "[EVAL] body_vy_mean  = " << g_state.sum_body_vy / n << " m/s" << std::endl;
    std::cout << "[EVAL] yaw_rate_mean = " << g_state.sum_yaw_rate / n << " rad/s" << std::endl;
    std::cout << "[EVAL] contact_frac  = " << g_state.sum_contact / n << std::endl;
    std::cout << "[EVAL] displacement  = (" << pelvis_pos[0] - g_state.start_x << ", "
              << pelvis_pos[1] - g_state.start_y << ") m" << std::endl;
    std::cout << "[EVAL] pelvis_z_min  = " << g_state.min_pelvis_z << " m" << std::endl;
    std::cout << "[EVAL] tau_ratio_max = " << g_state.max_torque_ratio << std::endl;
  }

  if (!headless)
  {
    mjv_freeScene(&scn);
    mjr_freeContext(&con);
  }
  mj_deleteData(d);
  mj_deleteModel(m);
  if (!headless)
  {
    glfwTerminate();
  }
  return 0;
}

void mouse_button_callback(GLFWwindow *window, int, int, int)
{
  button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
  button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
  button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
  glfwGetCursorPos(window, &lastx, &lasty);
}

void cursor_pos_callback(GLFWwindow *window, double xpos, double ypos)
{
  if (!button_left && !button_middle && !button_right)
  {
    return;
  }
  const int mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                         glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

  mjtMouse action = mjMOUSE_ZOOM;
  if (button_right)
  {
    action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  }
  else if (button_left)
  {
    action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  }

  mjv_moveCamera(m, action, MOUSE_SCALE * (xpos - lastx), MOUSE_SCALE * (ypos - lasty), &scn, &cam);
  lastx = xpos;
  lasty = ypos;
}

void scroll_callback(GLFWwindow *, double, double yoffset)
{
  mjv_moveCamera(m, mjMOUSE_ZOOM, 0.0, -ZOOM_SENSITIVITY * yoffset, &scn, &cam);
}
