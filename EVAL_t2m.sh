ln -s ../utils ./Evaluator_272/
ln -s ../humanml3d_272 ./Evaluator_272/
ln -s ../options ./Evaluator_272/
ln -s ../models ./Evaluator_272/
ln -s ../visualization ./Evaluator_272/
ln -s ../Causal_TAE ./Evaluator_272/
python eval_t2m.py --resume-pth Causal_TAE/net_last.pth --resume-trans /cpfs03/shared/IDC/wangjingbo_group/motionstreamer/Open_source_Train_AR_16_1024_fps_30_111M_9/latest.pth