import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.models.AlzheimerPredictor import AlzheimerPredictor

class TestAlzheimerPredictor:
    def test_predict(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [np.array([0.1, 0.2, 0.6, 0.1])]
        img_array = np.random.rand(1, 224, 224, 3)
        with patch('app.models.AlzheimerPredictor.get_model', return_value=mock_model):
            result = AlzheimerPredictor.predict(img_array)
            assert isinstance(result, list)
            assert len(result) == 4
            assert abs(sum(result) - 1.0) < 1e-6

    def test_invalid_input(self):
        with patch('app.models.AlzheimerPredictor.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.predict.side_effect = Exception('Invalid input')
            mock_get_model.return_value = mock_model
            with pytest.raises(Exception):
                AlzheimerPredictor.predict(None)

    def test_get_class_name(self):
        preds = [0.1, 0.2, 0.6, 0.1]
        class_name = AlzheimerPredictor.get_class_name(preds)
        assert class_name == 'NonDemented' 