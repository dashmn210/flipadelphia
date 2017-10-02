


# parse args

# use args to preprocess data
	# possibly cut by column into seerate files
	# check vocab
	# get vocab size
	# possibly rm out-of-vocab tokens


# if inference
	# restore all models in the spec from working_dir
# elif training
	# train all models in spec

# for each model in spec
	# inference on dataset
	# evaluate
		# categorical-specific
			# AUC
			# ROC curve
			# average feature correlation with this (cramer's V) 
		# continuous-specific
			# accuracy
			# MSE
			# residual plot
			# average feature correlation (point-biserial)
		# model-specific
			# conditional/marginal R^2
			# loss
			# attention dump? 
			# etc etc
