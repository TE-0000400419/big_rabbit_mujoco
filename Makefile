# =======================
# big_rabbit_mujoco Makefile
# =======================
# 使い方:
#   make               # ビルド（sim / test / dump ツール）
#   make test          # 関節<->モータ変換の単体テスト
#   make export        # Isaac の checkpoint からヘッダを生成
#   make verify        # クロス検証（obs54 / action10 / 参照テーブル）
#   make hold          # crouch 保持（段 4）
#   make tracking      # 追従性能の測定（段 5）
#   make walk          # 歩行の実測（段 6、5 条件）
#   make video         # 5 条件の mp4 を書き出す
#   make gui           # GUI で起動（DISPLAY が必要）
#   make foot          # 足裏の接触形状を確認（PNG）。FOOT_VIEW=1 でビューア
#   make check         # test -> verify -> hold -> tracking -> walk を通す
#   make clean
# =======================

BUILD_DIR   ?= build-linux
CMAKE       ?= cmake
PYTHON      ?= python3
ISAAC_ROOT  ?= /home/pomiou/work/big_rabbit_isaac
ISAAC_PYTHON?= /home/pomiou/work/env_isaaclab/bin/python

# 採用 policy。差し替えるときはここだけ変える。
POLICY_RUN    ?= big_rabbit_walk_v24_robust
POLICY_PARAMS ?= configs/big_rabbit_balance/big_rabbit_walk_v24_robust_4096_500.params.yaml
CHECKPOINT    ?= $(shell ls -t $(ISAAC_ROOT)/logs/rsl_rl/big_rabbit_balance/*_$(POLICY_RUN)/model_499.pt 2>/dev/null | head -1)

SIM        := $(BUILD_DIR)/big_rabbit_mujoco_sim
SCENE      := $(CURDIR)/robotmodel/big_rabbit/scene.xml
SCENE_TRACK := $(CURDIR)/robotmodel/big_rabbit/scene_tracking.xml
GAIT_PERIOD ?= 0.7
SIM_SECONDS ?= 11

.DEFAULT_GOAL := all
.PHONY: all configure test export verify hold tracking walk video gui foot check clean

all: configure
	$(CMAKE) --build $(BUILD_DIR) -j$(shell nproc)

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release

test: all
	./$(BUILD_DIR)/test_joint_motor_transform

# ---- Isaac の学習結果をヘッダへ落とす ----
export:
	@test -n "$(CHECKPOINT)" || (echo "checkpoint が見つからない。POLICY_RUN を確認する"; exit 1)
	$(ISAAC_PYTHON) tools/export_isaac_policy_headers.py \
	  --isaac-root $(ISAAC_ROOT) --checkpoint $(CHECKPOINT) --params $(POLICY_PARAMS)

# ---- クロス検証 段 1-3。obs54 / action10 / 参照テーブルが Isaac と一致するか ----
verify: all
	@test -n "$(CHECKPOINT)" || (echo "checkpoint が見つからない"; exit 1)
	$(PYTHON) tools/compare_with_isaac.py --isaac-root $(ISAAC_ROOT) --isaac-python $(ISAAC_PYTHON) \
	  --checkpoint $(CHECKPOINT) --params $(POLICY_PARAMS)

# ---- 段 4。crouch 保持 ----
hold: all
	BIG_RABBIT_SCENE_XML=$(SCENE) BIG_RABBIT_HEADLESS=1 BIG_RABBIT_CONTROL_MODE=home \
	BIG_RABBIT_MAX_SIM_TIME=3 BIG_RABBIT_LOG_INTERVAL_S=1 ./$(SIM)

# ---- 段 5。骨盤を空中に溶接して PD 追従だけを測る ----
tracking: all
	BIG_RABBIT_SCENE_XML=$(SCENE_TRACK) BIG_RABBIT_CONTROL_MODE=reference BIG_RABBIT_HEADLESS=1 \
	BIG_RABBIT_MAX_SIM_TIME=6 BIG_RABBIT_RL_START_S=0 BIG_RABBIT_LOG_INTERVAL_S=0 \
	BIG_RABBIT_MOTION_COMMAND_X=1 BIG_RABBIT_TRACE_CSV=$(CURDIR)/outputs/trace.csv ./$(SIM)
	$(PYTHON) tools/analyze_tracking.py $(CURDIR)/outputs/trace.csv --period $(GAIT_PERIOD)

# ---- 段 6。5 条件の実測 ----
walk: all
	@for spec in "zero 0 0" "forward 1 0" "backward -1 0" "turnleft 0 1" "turnright 0 -1"; do \
	  set -- $$spec; \
	  echo "##### $$1"; \
	  BIG_RABBIT_SCENE_XML=$(SCENE) BIG_RABBIT_HEADLESS=1 BIG_RABBIT_MAX_SIM_TIME=$(SIM_SECONDS) \
	  BIG_RABBIT_LOG_INTERVAL_S=0 BIG_RABBIT_MOTION_COMMAND_X=$$2 BIG_RABBIT_MOTION_COMMAND_YAW=$$3 \
	    ./$(SIM) | grep -a "EVAL\]"; \
	done

# ---- 動画。物理と制御は C++ 側が正本で、Python は qpos を描画するだけ ----
video: all
	@mkdir -p videos outputs
	@for spec in "zero 0 0" "forward 1 0" "backward -1 0" "turnleft 0 1" "turnright 0 -1"; do \
	  set -- $$spec; \
	  BIG_RABBIT_SCENE_XML=$(SCENE) BIG_RABBIT_HEADLESS=1 BIG_RABBIT_MAX_SIM_TIME=$(SIM_SECONDS) \
	  BIG_RABBIT_LOG_INTERVAL_S=0 BIG_RABBIT_MOTION_COMMAND_X=$$2 BIG_RABBIT_MOTION_COMMAND_YAW=$$3 \
	  BIG_RABBIT_QPOS_CSV=$(CURDIR)/outputs/qpos_$$1.csv ./$(SIM) > /dev/null; \
	  $(PYTHON) tools/render_qpos_video.py $(CURDIR)/outputs/qpos_$$1.csv --out videos/$$1.mp4; \
	done

gui: all
	BIG_RABBIT_SCENE_XML=$(SCENE) ./$(SIM)

# ---- 足裏の接触形状。group 3（衝突）を明示表示する。FOOT_VIEW=1 で対話ビューア ----
foot:
	@mkdir -p outputs
	$(PYTHON) tools/view_foot_collision.py --scene $(SCENE) --mode collision $(if $(FOOT_VIEW),--interactive,)
	@test -n "$(FOOT_VIEW)" || $(PYTHON) tools/view_foot_collision.py --scene $(SCENE) --mode overlay \
	  --out $(CURDIR)/outputs/foot_overlay.png

check: test verify hold tracking walk

clean:
	rm -rf $(BUILD_DIR)
