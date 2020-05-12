from cvfinetune.training.trainer import SacredTrainer


class SSDTrainer(SacredTrainer):

	def reportables(self, opts):
		print_values = [
			"elapsed_time",
			"epoch",

			"main/loss", self.eval_name("main/loss"),
			"main/loc_loss", self.eval_name("main/loc_loss"),
			"main/conf_loss", self.eval_name("main/conf_loss"),
		]

		plot_values = {
			"loss": [
				"main/loss", self.eval_name("main/loss"),
			],
			"loc_loss": [
				"main/loc_loss", self.eval_name("main/loc_loss"),
			],
			"conf_loss": [
				"main/conf_loss", self.eval_name("main/conf_loss"),
			],
		}
		return print_values, plot_values
