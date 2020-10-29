# demo here:
#!/usr/bin/env bash
fw_train_src_path=${1}
fw_train_tgt_path=${2}
fw_valid_src_path=${3}
fw_valid_tgt_path=${4}
save_dir=${5}


mkdir -p  ${save_dir}/fw
python preprocess.py \
--share_vocab \
-train_src ${fw_train_src_path} \
-train_tgt ${fw_train_tgt_path} \
-valid_src ${fw_valid_src_path} \
-valid_tgt ${fw_valid_tgt_path} \
-save_data ${save_dir}/fw/dataset


mkdir -p  ${save_dir}/bw
python preprocess.py \
--share_vocab \
-src_vocab ${save_dir}/fw/dataset.vocab.pt \
-train_src ${fw_train_tgt_path} \
-train_tgt ${fw_train_src_path} \
-valid_src ${fw_valid_tgt_path} \
-valid_tgt ${fw_valid_src_path} \
-save_data ${save_dir}/bw/dataset

cp ${save_dir}/bw/dataset.valid.0.pt ${save_dir}/fw/dataset.m_valid.0.pt