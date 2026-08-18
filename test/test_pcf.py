import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "learning"))

import pcf


def feasibility_data(input_dim=2, parameter_dim=2):
    return {
        "input_mean": jnp.zeros(input_dim),
        "input_std": jnp.ones(input_dim),
        "parameter_mean": jnp.zeros(parameter_dim),
        "parameter_std": jnp.ones(parameter_dim),
        "env_scale": 1.0,
        "gamma": 1.0,
        "parameter_dim": parameter_dim,
        "g_matrix": jnp.ones((1, input_dim)),
        "b_offset": jnp.ones(1) * 10.0,
        "b_theta": jnp.zeros((1, parameter_dim)),
    }


class TestPCF(unittest.TestCase):
    def setUp(self):
        self.input_dim = 2
        self.parameter_dim = 2
        self.params = pcf.init(
            jax.random.PRNGKey(0),
            input_dim=self.input_dim,
            parameter_dim=self.parameter_dim,
            convex_widths=(4,),
            hyper_widths=(5,),
        )
        self.model_input = jnp.ones((4, self.input_dim))
        self.parameter = jnp.ones((4, self.parameter_dim))
        self.y = jnp.ones(4)
        self.g = jnp.zeros((4, self.input_dim))
        self.w = jnp.ones(4)
        self.feasibility_data = feasibility_data(self.input_dim, self.parameter_dim)

    def test_init_metadata(self):
        self.assertEqual(self.params["input_dim"], self.input_dim)
        self.assertEqual(self.params["parameter_dim"], self.parameter_dim)
        self.assertNotIn("sections_static", self.params)
        self.assertEqual(len(self.params["sections"]), 3)

    def test_unpack_shapes(self):
        emitted = pcf._psi(self.parameter[0], self.params["hyper"])
        tensors = pcf.unpack(self.params["sections"], emitted)

        self.assertEqual(set(tensors), {"W", "V", "omega"})
        self.assertEqual(len(tensors["V"]), len(self.params["layer_dims"]) - 1)
        self.assertEqual(len(tensors["omega"]), len(self.params["layer_dims"]) - 1)
        self.assertEqual(len(tensors["W"]), len(self.params["layer_dims"]) - 2)

    def test_value_and_grad_shapes(self):
        values, grads = pcf.value_and_grad(
            self.params["hyper"],
            self.params["sections"],
            self.model_input,
            self.parameter,
        )

        self.assertEqual(values.shape, (4,))
        self.assertEqual(grads.shape, (4, self.input_dim))
        self.assertTrue(bool(jnp.all(jnp.isfinite(values))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(grads))))

    def test_pcf_class_methods(self):
        model = pcf.PCF(self.params)

        values = model(self.model_input, self.parameter)
        predicted = model.predict(self.model_input, self.parameter)
        grads = model.grad(self.model_input, self.parameter)

        np.testing.assert_allclose(np.asarray(values), np.asarray(predicted))
        self.assertEqual(grads.shape, (4, self.input_dim))

    def test_loss_returns_components(self):
        objective, parts = pcf.loss(
            self.params["hyper"],
            self.params["sections"],
            self.model_input,
            self.parameter,
            self.y,
            self.g,
            self.w,
            self.feasibility_data,
            feasibility_weight=0.0,
        )

        self.assertEqual(objective.shape, ())
        self.assertEqual(len(parts), 3)
        self.assertTrue(bool(jnp.isfinite(objective)))

    def test_train_one_epoch(self):
        trained = pcf.train(
            np.asarray(self.model_input),
            np.asarray(self.parameter),
            np.asarray(self.y),
            np.asarray(self.g),
            self.input_dim,
            self.parameter_dim,
            self.feasibility_data,
            convex_widths=(4,),
            hyper_widths=(5,),
            batch_size=2,
            epochs=1,
            eval_interval=1,
        )

        self.assertEqual(trained["input_dim"], self.input_dim)
        self.assertEqual(trained["parameter_dim"], self.parameter_dim)


if __name__ == "__main__":
    unittest.main()
