# 阶段 A — cage level 1（±8.6°/轴，步长 0.9 rad/s，夹爪 ±150 tick），增益保持 0.3
python examples/nexarm/teleoperate_pico.py --follower-port /dev/ttyUSB0 --no-camera --safety-level 1

# 阶段 B — level 1 + 增益提至 0.5（手部 2cm → EE 1cm，手感明显增强）
python examples/nexarm/teleoperate_pico.py --follower-port /dev/ttyUSB0 --no-camera \
    --safety-level 1 --pos-gain 0.5 --rot-gain 0.5

# 阶段 C — level 2（±23°/轴，步长 1.8 rad/s，夹爪 ±400 tick）+ 增益 0.7
python examples/nexarm/teleoperate_pico.py --follower-port /dev/ttyUSB0 --no-camera \
    --safety-level 2 --pos-gain 0.7 --rot-gain 0.7

# 阶段 D — 笼全开（全工作空间）+ 增益 1.0。⚠️ 臂可到达 URDF 全范围，确保周围无障碍
python examples/nexarm/teleoperate_pico.py --follower-port /dev/ttyUSB0 --no-camera \
    --safety-level 3 --pos-gain 1.0 --rot-gain 1.0

# 最佳实践  — 笼全开（全工作空间）增益提至 0.5（手部 2cm → EE 1cm，手感明显增强）
python examples/nexarm/teleoperate_pico.py --follower-port /dev/ttyUSB0 --no-camera \
    --safety-level 3 --pos-gain 0.5 --rot-gain kkk


    