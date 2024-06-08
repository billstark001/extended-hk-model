from functools import wraps
from inspect import signature, Parameter
from typing import Callable, Dict, Any, List, Optional, Union, Tuple


class Context:
  def __init__(self):
    self.state: Dict[str, Any] = {}
    self.selectors: Dict[str, Callable] = {}
    self.multi_selectors: Dict[Tuple[str, ...], Tuple[str, int]] = {}
    self.cache: Dict[str, Any] = {}

  def set_state(self, **state: Any) -> None:
    for key, value in state.items():
      self.state[key] = value
    self.cache = {}

  def add_selector(
      self,
      name: Union[str, Tuple[str, ...]],
      deps: List[str],
      func: Callable
  ) -> None:
    if isinstance(name, tuple):
      name_tuple = name
      name = str(name_tuple)
      for i, n in enumerate(name_tuple):
        if n in self.multi_selectors or n in self.selectors:
          raise ValueError(
              f"Multi-Selector '{name}' is already defined in the context")
        self.multi_selectors[n] = (name, i)
    if name in self.multi_selectors or name in self.selectors:
      raise ValueError(f"Selector '{name}' is already defined in the context")
    self.selectors[name] = (deps, func)
    
    
  def get(self, name: str) -> Any:
    if name in self.state:
      return self.state[name]
    if name in self.cache:
      return self.cache[name]
    if name in self.multi_selectors:
      selector_key, index = self.multi_selectors[name]
      ret = self.get(selector_key)
      return ret[index]
    if name not in self.selectors:
      raise ValueError(
          f"Selector '{name}' is not defined in the context")

    deps, func = self.selectors[name]
    dep_values = [self.get(dep) for dep in deps]
    value = func(*dep_values)
    self.cache[name] = value
    return value

  def get_values(self, *names):
    return [self.get(n) for n in names]

  def __getattr__(self, name: str) -> Any:
    return self.get(name)

  def selector(
      self,
      return_name: Union[None, str, Tuple[str, ...], List[str]] = None,
      deps_map: Optional[Dict[str, str]] = None
  ) -> Callable:

    def decorator(func: Callable) -> Callable:
      if not callable(func):
        raise TypeError(
            'Decorator applied on a non-callable object: ' + str(func))
      sig = signature(func)
      params = sig.parameters

      input_vars = [name for name, param in params.items()
                    if param.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)]

      # ensure it is a string tuple with length more than 2 or a string
      output_var = return_name if return_name else func.__name__
      if isinstance(output_var, list):
        output_var = tuple(output_var)
      if isinstance(output_var, tuple) and len(output_var) == 1:
        output_var = output_var[0]

      if deps_map:
        input_vars = [deps_map.get(name, name) for name in input_vars]

      self.add_selector(output_var, input_vars, func)
      return func

    # for usage `@context.selector`
    if callable(return_name):
      return decorator(return_name)

    # for usage `@context.selector()`
    return decorator


if __name__ == "__main__":
  context = Context()

  context.set_state(a=10, b=20)

  @context.selector(return_name='sum', deps_map=dict(x='a', y='b'))
  def c(x: int, y: int) -> int:
    return x + y

  @context.selector('double')
  def d(sum: int) -> int:
    return sum * 2
  
  @context.selector(('e', 'f'), deps_map=dict(c='sum', d='double'))
  def ef(c: int, d: int) -> int:
    return c + d, c - d

  print(context.double)
  print(context.sum)
  print(context.e)
  print(context.f)
