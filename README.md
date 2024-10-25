
# Differentiable Adaptive Merging (DAM)

![Project Figure](figures/readme.webp)

Differentiable Adaptive Merging (DAM)은 여러 LLM의 고유한 능력을 병합하여 데이터 효율성을 최적화하고 계산 비용을 줄이는 과정을 자동화합니다. DAM은 전통적인 방법과 진화적 방법보다 우수하며, 다목적 AI 시스템에 적합한 확장 가능한 솔루션을 제공합니다. 다양한 모델 병합 시나리오에서 DAM의 우수성이 광범위한 실험을 통해 입증되었습니다.

## 워크플로우 실행 단계

이 저장소에는 병합 계수를 튜닝하는 과정을 실행하기 위한 구현이 포함되어 있습니다.

### 1. 병합된 모델 생성
먼저, `dam` 폴더에 있는 `merge.py` 스크립트를 실행하여 병합된 모델을 생성합니다. 결과로 나온 병합된 모델은 학습되지 않은 계수를 포함하게 됩니다.

이 단계에서 사용자가 지정한 대로 각 모델의 레이어 노름, 임베딩 레이어 및 선형 레이어의 각 열에 학습 가능한 계수를 할당합니다. 이러한 계수는 학습 과정에서 최적의 모델 통합을 달성하기 위해 최적화됩니다.

#### 명령어:

```bash
python dam/merge.py mistralai/Mistral-7B-v0.1 augmxnt/shisa-gamma-7b-v1 WizardLM/WizardMath-7B-V1.1 arcee-train/Abel-7B-002-truncated-embeds --device cuda --output_path ./merged_model --repo_id arcee-train/[prefix]-untrained-merge
인자 설명:
base_model_id: 기본 모델의 ID. 이 모델의 모든 레이어는 DAM 레이어로 대체됩니다.
model_ids: 병합할 모델들의 ID. 이 모델들의 선형 레이어가 사용됩니다.
--output_path: 병합된 모델이 저장될 경로.
--device: 계산에 사용할 장치 (예: 'cpu', 'cuda').
--repo_id: 병합된 모델이 푸시될 저장소 ID.
--use_base_model: 지정되면 기본 모델의 선형 레이어에도 학습 가능한 계수가 추가됩니다. 이는 선택 사항입니다.
2. 데이터셋 준비
데이터셋을 준비하려면 dam/data 폴더로 이동하여 create_merge_dataset.py를 실행합니다. 이 스크립트는 병합하려는 모델들을 학습할 때 사용된 데이터에서 예제를 포함하는 복합 데이터셋을 생성하고, 이들의 템플릿을 적용하며, 데이터를 토크나이즈합니다. 선택적으로, 다른 모델에 대한 상위-K 로짓을 계산하여 저장할 수 있으며, 이는 나중에 학습 중에 사용됩니다. 또한 로짓을 미리 계산하는 것은 선택 사항입니다.

명령어:
bash
코드 복사
python dam/data/create_merge_dataset.py --dataset_names "p1atdev/ichikara-instruction:20231115-1,microsoft/orca-math-word-problems-200k,meta-math/MetaMathQA" --model_ids "augmxnt/shisa-gamma-7b-v1,WizardLM/WizardMath-7B-V1.1,arcee-train/Abel-7B-002-truncated-embeds" --base_model_name mistralai/Mistral-7B-v0.1 --cache_dir /home/ec2-user/.cache/huggingface --compute_logits True --dataset_id arcee-train/[prefix]-combined-dataset --example_count 1729 --max_length 2048 --add_top_k_logits False
인자 설명:
--dataset_names: 각 모델을 튜닝하는 데 사용된 데이터셋의 이름 목록. 각 데이터셋에서 샘플이 선택됩니다.
--base_model_dataset_name: 기본 모델 데이터셋의 이름. 이는 선택 사항입니다.
--model_ids: 로드할 모델들의 ID 목록.
--base_model_name: 기본 모델의 이름.
--cache_dir: 모델을 캐시할 디렉토리.
--compute_logits: True로 설정하면 상위-K 로짓을 계산하여 저장합니다. 선택 사항입니다.
--dataset_id: Hugging Face Hub에 푸시할 데이터셋 ID.
--example_count: 각 데이터셋에서 선택할 예제 수.
--max_length: 토크나이즈된 예제의 최대 길이.
--add_top_k_logits: 결합된 데이터셋에 상위-K 로짓을 추가합니다. 기본값은 False입니다.
--base_model_dataset_name: 기본 모델 데이터셋의 이름. 이는 선택 사항입니다.
3. 학습 실행
이 단계에서는 dam/train_dam.py 스크립트를 실행합니다. 이 단계의 목적은 병합 계수를 학습하는 것입니다. 학습이 완료되면 모델은 최적화된 계수와 함께 기본 모델 구조에 병합됩니다. 또한, 이 코드는 여러 GPU와도 호환됩니다.

train_dam.py 스크립트 상단에서 수동으로 설정을 구성할 수 있습니다.

손실 함수의 개별 구성 요소는 loss_fns 사전에서 True/False로 설정하여 활성화 또는 비활성화할 수 있습니다.
명령어:
bash
코드 복사
python dam/train_dam.py --learning_rate 1e-3 --lambda_coef_similarity 0.01 --generate_logits_on_fly True --untrained_merged_model_name arcee-train/[your-model] --combined_hf_dataset_dir arcee-train/[prefix]-combined-dataset --cache_dir /home/ec2-user/.cache/huggingface --base_model_name mistralai/Mistral-7B-v0.1 --use_wandb True
인자 설명:
--temperature: KL 발산을 위한 온도.
--weight_decay: 옵티마이저의 가중치 감쇠.
--learning_rate: 옵티마이저의 학습률.
--lr_scheduler_type: 학습률 스케줄러 유형 (linear 등).
--use_kl: 손실 함수에서 KL 발산 사용.
--use_mse: 손실 함수에서 평균 제곱 오차 사용.
--use_entropy: 손실 함수에서 엔트로피 사용.
--lambda_coef: 정규화를 위한 Lambda 계수.
--lambda_coef_l1: L1 정규화 계수.
--lambda_coef_l2: L2 정규화 계수.
--use_wandb: Weights and Biases에 학습 로그 업로드.
--generate_logits_on_fly: 학습 중에 로짓을 실시간으로 생성.
--use_all_logits: 학습 중 모든 로짓을 사용.
--untrained_merged_model_name: 학습되지 않은 병합 모델의 이름.
--combined_hf_dataset_dir: 로짓이 포함된 데이터셋의 디렉토리.
--cache_dir: 모델을 캐시할 디렉토리.
--base_model_name: 기본 모델의 이름.
인용
DAM 방법에 대한 논문이 있습니다. 다음과 같이 인용할 수 있습니다:


@article{gauthier2024merging,
  title={Merging in a Bottle: Differentiable Adaptive Merging (DAM) and the Path from Averaging to Automation},
  author={Gauthier-Caron, Thomas and Siriwardhana, Shamane and Stein, Elliot and Ehghaghi, Malikeh and Goddard, Charles and McQuade, Mark and Solawetz, Jacob and Labonne, Maxime},
  journal={arXiv preprint arXiv:2410.08371},
  year={2024}
}
