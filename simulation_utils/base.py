from inspect import signature


class BaseModel(object):
    """
    Base class for simulation models.

    """

    def get_params(self, deep=True):
        """
        Get the parameters for this model.

        Args:
            deep (bool): whether or not to return the parameters of
            nested models.

        Returns:
            dict: The model parameters

        """
        params = dict()
        sig = signature(self.__init__)
        for param in sig.parameters.values():
            pname = param.name
            value = getattr(self, pname, None)
            params[pname] = value
            if deep and hasattr(value, 'get_params'):
                nested_params = value.get_params()
                for k, v in nested_params.items():
                    nested_pname = ''.join([pname, '__', k])
                    params[nested_pname] = v
        return params

    def set_params(self, **kwargs):
        """
        Set parameters for this model.

        The parameters should match the parameters of the initialization.

        Args:
            **kwargs: Keywords arguments for parameters to be set.

        Returns:
            self: The model with updated parameters.

        """
        params = self.get_params()
        # sort items by keys in order to set the nested model before its params.
        for k, v in sorted(kwargs.items(), key=lambda x: x[0]):
            pname, _, next_pname = k.partition('__')
            if pname not in params:
                raise ValueError(f'Invalid parameter {pname} for model {self}')

            if next_pname and hasattr(params[pname], 'set_params'):
                next_param = {next_pname: v}
                params[pname].set_params(**next_param)
            else:
                setattr(self, pname, v)
                params[pname] = v  # update view dict.
        return self
