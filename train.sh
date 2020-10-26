# demo here:
## forward pretraining
python train.py \
-train_type="pretrain" \
-data data/fw/dataset \
-save_model outputs/s1/

## backward pretrainings
python train.py \
-train_type="pretrain" \
-data data/bw/dataset \
-save_model outputs/s2/

## noise pretrainings
export FORWARD_TRAINED_STEP=100000
export BACKWARD_TRAINED_STEP=100000
python train.py \
-train_type="nmt_noise" \
-data data/fw/dataset \
--g_train_from outputs/s1/_step_${FORWARD_TRAINED_STEP}.pt \
--m_train_from outputs/s2/_step_${BACKWARD_TRAINED_STEP}.pt \
-save_model outputs/s3/ \
--learning_rate=0.01  

## adversarial training
export FORWARD_TRAINED_STEP=100000
export BACKWARD_TRAINED_STEP=100000
export NOISE_TRAINED_STEP=10000
python train.py \
-train_type="nmt_finetune" \
-data data/fw/dataset \
--g_train_from outputs/s1/_step_${FORWARD_TRAINED_STEP}.pt \
--m_train_from outputs/s2/_step_${BACKWARD_TRAINED_STEP}.pt \
--n_train_from outputs/s3/n_model/-${NOISE_TRAINED_STEP}.pt \
--save_model ../outputs/wh/ \
--learning_rate=0.01 \
--max_grad_norm 0.1






