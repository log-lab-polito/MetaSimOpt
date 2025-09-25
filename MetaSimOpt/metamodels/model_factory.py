import inspect

class ModelFactory:

    def __init__(self, model_class, **default_args):
        self.model_class = model_class

        # Recupera gli hyperparametri se disponibili
        if 'hyperparameters' not in default_args and hasattr(model_class, 'get_model_hyperparameters'):
            default_args['hyperparameters'] = model_class.get_model_hyperparameters()

        self.default_args = default_args
        self.required_args, self.accepted_args = self._get_model_args()

    def _get_model_args(self):
        """
        Recupera gli argomenti richiesti e accettati dal costruttore del modello
        """

        sig = inspect.signature(self.model_class.__init__)
        required = []
        accepted = []

        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            accepted.append(name)
            if param.default == inspect.Parameter.empty:
                required.append(name)

        return required, accepted

    def create(self, **overrides):
        """Crea un'istanza del modello, validando e filtrando gli argomenti"""
        # Unione: default + override
        combined_args = self.default_args.copy()
        combined_args.update(overrides)

        # Validazione: controlla che tutti gli argomenti richiesti siano presenti
        missing = [arg for arg in self.required_args if arg not in combined_args]
        if missing:
            raise ValueError(f"Missing required arguments for model '{self.model_class.__name__}': {missing}")

        # Filtro: passa solo quelli accettati
        final_args = {k: v for k, v in combined_args.items() if k in self.accepted_args}

        return self.model_class(**final_args)
