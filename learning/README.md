# 파란학기 통합 학습 파이프라인

VR Meta Quest로 팔과 ORCA 손을 함께 시연하고, 같은 데이터로 ACT와 π0.5를 학습하는 공통 경로입니다. ROS2는 사용하지 않습니다.

현재 구현 기준은 이 저장소의 **AM-ARM200 팔 3관절 + ORCA 오른손 17관절 = 20차원**입니다. SO-101 팔 5관절을 사용하는 22차원 로봇을 구현한 것으로 간주하면 안 됩니다. SO-101로 바뀌면 팔 드라이버·기구학·joint map·캘리브레이션을 함께 교체하고 데이터를 새로 검증해야 합니다.

## 1. 연결 구조

```text
Meta Quest 손목 위치 ── 팔 IK ──────────┐
Meta Quest 손가락/손목 회전 ─ 리타게팅 ─┤
                                       ↓
                         공통 20D 모델 관절 목표 [rad]
                                       ↓
                         관절 한계·속도·추종 오차 검사
                                       ↓
                         AM-ARM200 + ORCA 드라이버
                                       ↓
Top / Front / Wrist 영상 + 측정 state + 전송 action
                                       ↓
                        원본 시연 보존 + 품질 검사
                                       ↓
                  같은 에피소드 분할의 LeRobot 데이터
                       ┌───────────────┴─────────────┐
                   ACT 학습                     π0.5 미세조정
                       └───────────────┬─────────────┘
                                       ↓
                 저장된 정규화 복원 → 20D 관절 목표 → 안전 검사
                                       ↓
                          동일 드라이버로 실행 (추가 IK 없음)
```

풋 페달은 에피소드 상태만 변경합니다. 로봇 통신은 하나의 제어 루프가 담당하고, 카메라·VR 처리·원본 저장은 별도 작업으로 분리했습니다.

## 2. 지금 검증된 범위

- 로봇/인터넷/GPU 없는 자동 테스트 **45개 통과**.
- 모의 시연 3개 × 30프레임 = **90프레임** 수집 및 무결성 검사 통과.
- LeRobot v0.4.4 공개 API에 맞춘 변환기, ACT/π0.5 학습 실행기, 체크포인트 실행기, 오프라인 평가 명령 구현.
- 데이터 변환기의 API 호출·분할·통계와 학습 명령은 테스트 대역으로 검증했습니다.
- **실제 LeRobot 비디오/Parquet 변환·모델 forward/backward·실기 동작·Quest 연결은 아직 실행하지 않았습니다.** 현재 작업 PC에는 torch/LeRobot/MuJoCo/OpenCV/evdev가 없습니다.
- 새 hardware 설정은 기본 잠금 상태입니다. 코드가 있다고 안전 인증이나 집기 성공이 확보된 것은 아닙니다.

## 3. 장비 없이 먼저 실행

저장소 최상위에서 Python 3.10–3.12를 사용합니다. 가상환경 활성화 후:

```bash
python -m pip install -e ./learning
python scripts/paran_pipeline.py doctor
python scripts/paran_pipeline.py schema
python -m unittest discover -s learning/tests -v
python scripts/paran_pipeline.py record --backend mock --raw-root learning_outputs/mock_raw --mock-episodes 3 --frames 30 --fast
python scripts/paran_pipeline.py inspect --raw-root learning_outputs/mock_raw --allow-synthetic --output learning_outputs/mock_report.json
```

`--fast`는 모의 실험 전용입니다. 실제 시간 간격 검증에는 `--fast`를 빼세요. 반복 실행 시 결과·보고서에는 새로운 경로를 사용합니다. 모의 데이터는 `synthetic: true`로 표시되며 실데이터나 실기 정책에 자동으로 섞이지 않습니다.

## 4. 로봇/학습 환경 설치

풋 페달을 직접 읽는 수집 PC는 Linux를 대상으로 합니다. Windows에서는 모의 경로와 터미널의 n/r/q 입력을 사용할 수 있지만 실제 장비 경로는 미검증입니다. 기존 작업 환경을 덮어쓰지 말고 별도 가상환경을 만드세요.

수집 환경(저장소 최상위):

```bash
python -m pip install -e ./orca_core
python -m pip install -r learning/requirements-robot.txt
python -m pip install --no-deps -e ./orca_teleop
python -m pip install -e "./learning[pedal]"
python -m pip check
```

여기서 `--no-deps`는 팀원이 수정한 **로컬 orca_core**가 upstream `main` 버전으로 교체되지 않게 하기 위한 것입니다. 필요한 teleop 의존성은 앞의 requirements에서 설치합니다. 기존 전용 설치 절차가 있다면 로컬 소스가 import되는지 확인한 뒤 재사용할 수 있습니다.

학습 환경(로봇 없이 별도 GPU PC에서도 가능):

```bash
python -m pip install -e "./learning[learning]"
python -m pip check
```

LeRobot은 **0.4.4로 고정**합니다. 해당 버전의 `pi` extra는 전용 Transformers 브랜치를 사용하므로 일반 Transformers로 임의 교체하지 마세요. 실제 설치가 완료된 환경의 `pip freeze`도 실험 기록에 보관하세요. 전체 하위 의존성의 재현 가능한 lockfile은 GPU/OS 환경을 확정한 뒤 생성해야 합니다. FFmpeg/비디오 디코더 설치 및 CUDA·torch·torchvision·torchcodec 조합도 해당 PC에서 확인해야 합니다.

로봇 PC에서 정책까지 실행하려면 두 환경의 필요 패키지가 함께 설치되어야 합니다. π0.5 사전학습 가중치 및 PaliGemma tokenizer 접근 권한, GPU 메모리, 추론 지연은 별도 준비 사항입니다. 이 작업에서 가중치 다운로드나 GPU 작업은 실행하지 않았습니다.

## 5. 실기 설정: 반드시 먼저 확인

`configs/learning/hardware.template.json`을 새 `hardware.local.json`으로 복사해 작성합니다. 빈 배열은 의도적인 잠금 장치이며 임의의 큰 범위로 채우면 안 됩니다.

1. 실제 조립체가 AM-ARM200 3축 + ORCA 17축인지 확인합니다. SO-101이라면 여기서 중단하고 모델을 변경합니다.
2. 팔·손을 물리적으로 지지하고, 비상정지·낙하 방지·작업 공간 충돌 방지 절차를 마련합니다. 연결 초기화 중 토크가 해제될 수 있습니다.
3. ORCA의 실제 `port`, `motor_type: feetech`, `baudrate`, 손 캘리브레이션을 먼저 확정합니다. 학습 모드에서는 자동 포트 탐색·자동 손 캘리브레이션·자동 중립 이동을 금지합니다.
4. 팔 YAML에 고정된 `motor_zero_rad` 3개와 `boot_model_limits_rad`의 3×2 범위를 추가합니다. 순서는 shoulder_pan, shoulder_lift, elbow_flex입니다. 연결할 때마다 현재 위치를 0으로 다시 잡으면 안 됩니다. `max_step_from_home_rad`는 null로 두고 새 안전 필터에서 한계를 관리합니다.
5. `schema` 출력 순서로 20개씩 `lower_rad`, `upper_rad`, `max_velocity_rad_s`, `max_tracking_error_rad`, `episode_start_rad`, `episode_start_tolerance_rad`를 작성합니다. joint map의 팔 ±3.14는 물리 안전 범위가 아닙니다.
6. operator calibration 파일과 실제 로봇 캘리브레이션 ID를 연결합니다. 작업자 파일은 로봇 영점과 다른 개념입니다. 파일 해시는 시연에 보존합니다.
7. top/front/wrist의 장치 번호·해상도를 고정합니다. Linux `/dev/v4l/by-id/...` 경로도 사용할 수 있습니다. 기준값은 카메라 30Hz, 기록 15Hz, 640×480이며 USB 대역폭·노출·실제 처리 시간을 측정해서 조정해야 합니다.
8. 위 검토가 끝나야 `hardware_reviewed: true`와 실행 시 `--enable-hardware`를 함께 사용합니다.

종료 시 새 학습 모드는 **마지막 모터 목표와 토크 상태를 유지하는 방향으로 통신을 닫습니다**. 자동 torque-off로 물체/팔이 떨어지지 않게 하려는 정책이지, 안전한 정지를 보장한다는 뜻은 아닙니다. 통신 장애 시 모터가 무엇을 유지하는지는 펌웨어에 의존합니다. 장비를 물리적으로 받친 후 담당자가 비상정지/전원 차단 절차를 수행해야 합니다. 풋 페달이나 PC 프로그램을 E-stop으로 사용하지 마세요.

## 6. VR 시연 수집과 풋 페달

실기 준비 완료 후에만:

```bash
python scripts/paran_pipeline.py record --backend hardware --hardware-config configs/learning/hardware.local.json --enable-hardware --raw-root learning_outputs/real_raw --pedal-device /dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd
```

이 명령은 소유한 Quest 웹 publisher도 시작합니다. 기본 웹 포트는 8443입니다. 기존 Quest WebXR 접속/인증서 절차로 수집 PC에 접속하세요. 같은 포트에 기존 publisher를 중복 실행하면 안 됩니다. 학습 수집은 에피소드별 손목 기준점 초기화를 위해 이 소유 publisher를 사용합니다. 같은 PC의 옛 `record_dataset.py`나 별도 robot worker를 동시에 실행하지 마세요.

| 상태 | 오른쪽 KEY_C / n | 왼쪽 KEY_A / r | 가운데 KEY_B 두 번 / q |
|---|---|---|---|
| 준비·재설정 | 시작 자세 확인 후 녹화 시작 | 비녹화 VR 조작 켜기 | 세션 종료 |
| 비녹화 VR 조작 | 시작 자세 확인 후 녹화 시작 | 목표 갱신 정지 | 세션 종료 |
| 녹화 | 에피소드 저장 후 정지 | 거절 표시 후 정지 | 미완료로 보존 후 종료 |

키보드는 n/r/q를 입력하고 Enter를 누릅니다. 페달 키 반복·짧은 바운스는 무시하고 가운데는 2초 안의 두 번 누름을 확인합니다. 기존 풋 페달 UInput 브리지는 함께 실행하지 않습니다. 직접 장치를 읽기 때문에 창 포커스에 의존하지 않지만 evdev 접근 권한이 필요합니다.

매 회 물체 배치·팔 시작 자세·작업자 손목 중립 자세를 확인하세요. 오른쪽 페달 후 즉시 움직이지 않고, Quest 손목·IK 기준·리타게터를 재설정하고 신선한 입력을 확인합니다. 녹화 종료 후 자동 homing은 하지 않습니다. 왼쪽을 눌러 비녹화 VR 조작으로 시작 자세에 복귀한 다음 다시 녹화합니다. 클러치 해제는 팔만 멈추고 손가락은 계속 움직일 수 있습니다.

## 7. 저장 형식과 품질 기준

원본은 `raw_root/날짜_UUID/manifest.json` 및 `frame_000000.npz` 형태입니다. 삭제 대신 accepted/rejected/incomplete로 구분합니다. 중단된 원본과 이미 쓴 프레임은 남습니다. 완료된 원본을 이어쓰기/덮어쓰기 하지 않습니다.

- `state`: 해당 제어 시점에 읽은 드라이버의 20D 관절 상태. ORCA 손끝의 독립적인 실제 관절 센싱을 뜻하지 않습니다.
- `proposed_action`: VR→IK/리타게팅 또는 정책에서 제안한 관절 목표.
- `action`: 관절·속도 제한 적용 후 드라이버 전송 호출이 성공 반환한 목표. 실제 모터 도착/패킷 수신 ACK를 뜻하지 않습니다. 팔과 손 버스 전송은 원자적이지 않으며 중간 실패는 에피소드 전체 실패로 처리합니다.
- 3개 RGB 영상, 각 카메라 시퀀스/수신 시간, 상태 읽기 시작·끝, 제어 결정·전송 시작·끝, 선택된 VR 원본 keypoints/손목 transform/클러치 정보.
- 같은 PC의 단조 증가 고해상도 시계 `perf_counter_ns`를 사용합니다. **카메라 수신 시간은 노출 시간이 아닙니다.** 센서 하드웨어 동기화와 통신 지연 보정이 구현된 것은 아닙니다.
- VR와 카메라의 모든 원본 패킷을 저장하는 것은 아니며, 실제 제어에 선택된 시점의 원본을 저장합니다.

기준 검사는 입력 나이 120ms, 카메라 수신 시각 차이 50ms, 중복 카메라 프레임 금지, 일정한 영상 크기, 유한한 20D 벡터입니다. 제어 루프가 40ms 이상 늦거나 직렬 전송이 한 제어 주기를 초과하면 실패 처리합니다. 디스크 큐가 차도 프레임을 조용히 버리지 않습니다. 검증된 실행 주파수를 낮추거나 장비 문제를 해결하고 다시 수집해야 합니다.

현재 원본 RGB는 무압축 보존합니다. 640×480 RGB 3대, 15Hz이면 영상만 대략 **41MB/s, 2.5GB/분**이므로 SSD 쓰기속도와 여유 공간을 먼저 측정하세요. 이 용량은 압축된 LeRobot 비디오의 크기와 다릅니다.

## 8. LeRobot 데이터 변환

실제 학습 환경에서:

```bash
python scripts/paran_pipeline.py inspect --raw-root learning_outputs/real_raw --output learning_outputs/real_report.json
python scripts/paran_pipeline.py export --raw-root learning_outputs/real_raw --output learning_outputs/dataset_v1
```

accepted만 사용하며 최소 3개 에피소드가 있어야 합니다. 3개는 변환 검사용 최소치이지 충분한 학습량이 아닙니다. 에피소드 단위로 대략 80/10/10을 분할하고, train/validation/test를 별도 LeRobotDataset으로 생성합니다. 두 모델은 동일한 `paran_manifest.json`을 사용합니다. 전 프레임에서 state/action 평균·표준편차·q01/q99를 계산하되 학습에는 train 통계만 들어갑니다. 녹화 시간은 검사 후 규칙적인 학습 인덱스 frame_index/fps에 대응시키고 실제 원본 시간은 별도로 유지합니다. 큰 시간 공백을 임의 보간하지 않습니다.

변환 대상 경로가 있으면 중단합니다. 변환 실패 후 일부 출력이 남아도 manifest가 complete가 아니므로 학습 실행기가 거절합니다. 새 경로로 재변환하세요. 모의 경로 검증에는 별도 데이터 루트와 `--allow-synthetic`을 명시합니다.

## 9. ACT와 π0.5 학습

먼저 명령만 확인합니다. 아래 명령은 기본적으로 학습을 실행하지 않습니다.

```bash
python scripts/paran_pipeline.py train --dataset learning_outputs/dataset_v1 --model act --output learning_outputs/act_v1 --steps 20000
python scripts/paran_pipeline.py train --dataset learning_outputs/dataset_v1 --model pi05 --output learning_outputs/pi05_v1 --steps 20000
```

GPU/권한/데이터를 확인한 뒤 `--execute`를 붙여 실행합니다. 처음에는 두 모델 모두 새 출력 경로에서 `--steps 2`로 실제 배치 로딩→forward→backward→체크포인트 저장을 시험하세요. ACT 기본 batch 8, π0.5 기본 batch 1은 출발점일 뿐이며 GPU 조건에 맞게 `--batch-size`를 조정합니다.

- ACT: 영상 3대 + 20D state → 20D 관절 목표 청크.
- π0.5: 영상 3대 + state + task 문장 → 같은 20D 관절 목표 청크. `lerobot/pi05_base`에서 미세조정합니다. task의 객체·행동·목표 위치를 시연과 일치시키세요.
- 둘 다 chunk_size=50, n_action_steps=10으로 출발합니다. 15Hz에서 10개 실행은 약 0.67초이므로 얇은 물체 접촉 조작에는 실행 청크 길이·빈도를 별도 검증해야 합니다.
- π0.5 내부 최대 차원 32는 모델 내부 표현입니다. 로봇의 state/action을 32축으로 꾸며 저장하지 않습니다.
- 자동 Hub 업로드·W&B 전송은 꺼져 있습니다. 사전학습 모델 다운로드는 실행 시 필요할 수 있습니다.
- 성공적으로 종료된 학습 체크포인트에는 `paran_contract.json`을 함께 저장합니다. 중단된 학습의 체크포인트를 임의로 실기에 사용하지 마세요.

## 10. 평가와 정책 실행

```bash
python scripts/paran_pipeline.py evaluate --dataset learning_outputs/dataset_v1 --checkpoint learning_outputs/act_v1/checkpoints/last/pretrained_model --output learning_outputs/act_validation.json --device cuda
```

π0.5도 체크포인트만 바꿉니다. 기본 validation, 최종 확정 후에만 `--split test`를 사용하세요. 이 값은 모델별 정규화된 학습 목적의 손실입니다. ACT 손실과 π0.5 손실을 그대로 비교해 집기 성공률이라고 해석하면 안 됩니다.

실기 평가 승인을 별도로 받은 후:

```bash
python scripts/paran_pipeline.py run-policy --backend hardware --hardware-config configs/learning/hardware.local.json --enable-hardware --checkpoint learning_outputs/act_v1/checkpoints/last/pretrained_model --device cuda --pedal-device /dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd
```

정책과 저장된 전처리·역정규화를 로봇 연결 전에 로드하고, 관절 순서·차원·카메라·캘리브레이션 계약이 다르면 거절합니다. 모의 학습 정책은 실기에 사용할 수 없습니다. 오른쪽 페달로 실행/정지합니다. 정책 실행은 시연 데이터에 자동 저장하지 않습니다. 모델의 `select_action()`이 청크 큐를 관리하고 초기화 때 큐를 비웁니다. 출력에는 IK를 다시 적용하지 않습니다.

현재 실행기는 **동기 추론**입니다. 특히 π0.5가 시간 예산을 넘으면 멈춥니다. 별도 GPU 추론 서버·비동기 청크 연결·RTC는 구현하지 않았습니다. 실제 추론 지연 측정 후 필요하면 다음 단계로 추가해야 합니다. 정책 실행 시에는 VR preview가 없으므로 앞선 시연 수집의 비녹화 조작으로 시작 자세를 마련하세요.

실기 비교는 같은 미학습 물체 배치에서 집기/들기/지정 위치 내려놓기 성공, 중복 집기, 미끄러짐, 충돌, 완료 시간과 개입 횟수를 수동 기록하세요. 두 모델에 같은 시연·같은 조건·같은 판정 기준을 적용합니다.

## 11. 팀의 다음 확인 순서

1. 조립체와 20D/22D 여부 확정, 팔 고정 영점·ORCA 캘리브레이션·안전 범위 작성.
2. 세 카메라 USB/노출/장착, 모포·물체·목표 영역 고정, 실제 15Hz 수집 지속 가능 여부 측정.
3. 페달·Quest·손목 중립/클러치 초기화와 비녹화 조작 점검, 짧은 실제 시연 3개 검증.
4. 실데이터 변환 및 ACT/π0.5 각각 2-step 실제 학습 smoke test.
5. ACT 기준선 확보 후 π0.5 비교. 학습량·성공률은 실험으로 판단하고 미리 보장하지 않기.

## 출처와 변경 범위

- [saybeok/orca-hand](https://github.com/saybeok/orca-hand), 시작 커밋 `e3e9290830f7bf5e84cb64b907690fc2c1f19783`의 기존 팔 IK·ORCA 리타게터·하드웨어 인터페이스 재사용.
- [ValentinJaeung/lerobot-footpedal](https://github.com/ValentinJaeung/lerobot-footpedal), 커밋 `c215176d3f8530e76d482f665544c1b3e312ff42`의 매핑·디바운스·가운데 두 번 누르기 방식 반영. 원 저장소는 변경하지 않았고 MIT 고지는 `THIRD_PARTY/lerobot-footpedal.LICENSE`에 보존했습니다.
- [LeRobot v0.4.4 Dataset](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/datasets/lerobot_dataset.py), [정책/processor factory](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/factory.py), [π0.5 processor](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/pi05/processor_pi05.py)의 실제 API를 기준으로 작성했습니다.

기존 teleop entrypoint는 유지하고 `scripts/paran_pipeline.py`를 추가했습니다. 드라이버 변경은 새 학습 모드에서 엄격한 통신 실패 검사, 고정 팔 영점, 시작/종료 정책을 적용하기 위한 것입니다. 팀 작업은 `paran` 브랜치에서 진행하고, 검토된 변경만 PR로 `main`에 반영합니다. 원본 데이터·개인 설정은 커밋하지 않습니다.
