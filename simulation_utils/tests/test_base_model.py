from simulation_utils.base import BaseModel

init_params = {'str_param': 'simple', 'num_param': 1,
               'list_param': [1, 2], 'dict_param': {'key': 1}}

outer_params = {'outer_str': 'nested', 'outer_num': 2,
                'outer_dict': {'key': 2}}


class SimpleModel(BaseModel):
    def __init__(self, str_param, num_param, list_param, dict_param):
        self.str_param = str_param
        self.num_param = num_param
        self.list_param = list_param
        self.dict_param = dict_param


class NestedModel(BaseModel):
    def __init__(self, outer_str, outer_num, outer_dict, inner_model):
        self.outer_str = outer_str
        self.outer_num = outer_num
        self.outer_dict = outer_dict
        self.inner_model = inner_model


def test_get_params():
    test_model = SimpleModel(**init_params)
    assert test_model.get_params() == init_params


def test_get_params_nested_model():
    inner_model = SimpleModel(**init_params)
    outer_model = NestedModel(**outer_params, inner_model=inner_model)
    # test deep get_params
    expected_deep = {'outer_str': 'nested', 'outer_num': 2,
                     'outer_dict': {'key': 2}, 'inner_model': inner_model,
                     'inner_model__str_param': 'simple',
                     'inner_model__num_param': 1,
                     'inner_model__list_param': [1, 2],
                     'inner_model__dict_param': {'key': 1}}
    assert outer_model.get_params(deep=True) == expected_deep
    # test get_params without deep
    expected = {'outer_str': 'nested', 'outer_num': 2,
                'outer_dict': {'key': 2}, 'inner_model': inner_model}
    assert outer_model.get_params(deep=False) == expected


def test_set_params():
    test_model = SimpleModel(**init_params)
    kwargs = {'str_param': 'new_model', 'num_param': 22,
              'list_param': [3, 2], 'dict_param': {'j': 2, 'k': 3}}
    test_model.set_params(**kwargs)
    assert test_model.get_params() != init_params
    assert test_model.get_params() == kwargs


def test_set_params_nested_model():
    inner_model = SimpleModel(**init_params)
    outer_model = NestedModel(**outer_params, inner_model=inner_model)

    # test setting a new inner model
    new_inner_params = {'str_param': 'new_inner_model', 'num_param': 3,
                        'list_param': [1, 2, 3], 'dict_param': {'key': 3}}
    new_inner_model = SimpleModel(**new_inner_params)
    new_nested_params = {'outer_str': 'new_nested', 'outer_num': 3,
                         'outer_dict': {'key': 4},
                         'inner_model': new_inner_model}
    outer_model.set_params(**new_nested_params)
    expected = {'outer_str': 'new_nested', 'outer_num': 3,
                'outer_dict': {'key': 4}, 'inner_model': new_inner_model,
                'inner_model__str_param': 'new_inner_model',
                'inner_model__num_param': 3,
                'inner_model__list_param': [1, 2, 3],
                'inner_model__dict_param': {'key': 3}}
    actual = outer_model.get_params(deep=True)
    assert actual == expected

    # test setting new parameters for the new inner model
    new_inner_model = SimpleModel(**init_params)
    new_nested_params.update({'inner_model': new_inner_model,
                              'inner_model__str_param': 'new_inner_model',
                              'inner_model__num_param': 5,
                              'inner_model__list_param': [1, 2, 3, 4],
                              'inner_model__dict_param': {'key': 4}})
    outer_model.set_params(**new_nested_params)
    actual = outer_model.get_params(deep=True)
    assert actual == new_nested_params
