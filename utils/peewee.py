from typing import Dict, Type, Literal, Any, Optional, Union
import io
import peewee
from playhouse.migrate import SqliteMigrator, migrate
import numpy as np


def nullable(exclude=None):
  if exclude is None:
    exclude = []

  def decorator(cls):
    for key, value in cls.__dict__.items():
      if (
          isinstance(value, (peewee.Field, peewee.FieldAccessor))
          and not getattr(value, 'primary_key', False)
          and key not in exclude
      ):
        if isinstance(value, (peewee.Field)):
          value.null = True
        else:
          value.field.null = True
    return cls
  return decorator


class NumpyArrayField(peewee.BlobField):
  """
  支持自动序列化/反序列化 numpy.ndarray 的 Peewee 字段。
  支持 list 自动转 np.ndarray。
  支持可选强制类型（force_dtype），在读写时用 astype。
  """

  def __init__(self, dtype: Optional[Union[str, np.dtype]] = None, *args, **kwargs):
    """
    :param force_dtype: 强制类型（如np.float32），None则不转换。
    其余参数同peewee.BlobField
    """
    self.force_dtype = np.dtype(
        dtype) if dtype is not None else None
    super().__init__(*args, **kwargs)

  def _to_numpy(self, value: Any) -> np.ndarray:
    # list -> ndarray
    if isinstance(value, np.ndarray):
      arr = value
    elif isinstance(value, list):
      try:
        arr = np.array(value)
      except Exception as e:
        raise ValueError(f"Cannot convert list to ndarray: {e}")
    else:
      raise ValueError(
          f"Unsupported value type for NumpyArrayField: {type(value)}")
    # 类型转换
    if self.force_dtype is not None:
      try:
        arr = arr.astype(self.force_dtype)
      except Exception as e:
        raise ValueError(
            f"Failed to convert array to dtype {self.force_dtype}: {e}")
    return arr

  def db_value(self, value: Optional[Any]) -> Optional[bytes]:
    # None 直接存null
    if value is None:
      return None
    # 支持list自动转换
    arr = self._to_numpy(value)
    with io.BytesIO() as buf:
      np.save(buf, arr, allow_pickle=False)
      return buf.getvalue()

  def python_value(self, value: Optional[bytes]) -> Optional[np.ndarray]:
    if value is None:
      return None
    with io.BytesIO(value) as buf:
      arr: np.ndarray = np.load(buf, allow_pickle=False)
    # 强制类型转换
    if self.force_dtype is not None:
      try:
        arr = arr.astype(self.force_dtype)
      except Exception as e:
        raise ValueError(
            f"Failed to convert loaded array to dtype {self.force_dtype}: {e}")
    return arr


def sync_peewee_table(
    db: Optional[peewee.Database],
    model: Type[peewee.Model],
    extra_columns: Literal['ignore', 'error', 'delete'] = 'ignore',
    notnull_sync: Literal['ignore', 'error', 'warn', 'fix'] = 'fix',
) -> peewee.Database | None:
  import warnings

  db_is_none = db is None
  if db_is_none:
    db = model._meta.database  # type: ignore
  table_name: str = model._meta.table_name  # type: ignore

  # 1. 获取模型字段名
  model_fields: Dict[str, Any] = model._meta.fields.copy()  # type: ignore
  model_field_names = set(model_fields.keys())

  # 2. 获取数据库字段名
  assert db is not None
  db.connect(reuse_if_open=True)
  cursor = db.execute_sql(f'PRAGMA table_info("{table_name}");')
  # row[1]: column name
  db_columns = {row[1]: row for row in cursor.fetchall()}
  db_field_names = set(db_columns.keys())

  # 3. 新增字段
  migrator = SqliteMigrator(db)
  operations = []
  for field_name in model_field_names - db_field_names:
    field = model_fields[field_name]
    operations.append(migrator.add_column(table_name, field_name, field))

  # 4. 多余字段
  extra_db_fields = db_field_names - model_field_names
  if extra_db_fields:
    if extra_columns == 'error':
      raise ValueError(f"Extra columns in the table: {extra_db_fields}")
    elif extra_columns == 'delete':
      for field_name in extra_db_fields:
        operations.append(migrator.drop_column(table_name, field_name))

  # 5. 检查并同步 NOT NULL 约束
  mismatched_notnull = []
  for field_name in model_field_names & db_field_names:
    model_field = model_fields[field_name]
    db_col_info = db_columns[field_name]
    model_notnull = not getattr(model_field, 'null', False)
    db_notnull = bool(db_col_info[3])
    if model_notnull != db_notnull:
      mismatched_notnull.append((field_name, model_notnull, db_notnull))

  if mismatched_notnull:
    msg = (
        "Fields with mismatched NOT NULL constraint: " +
        ", ".join([
            f"{field} (model: {model_nn}, db: {db_nn})"
            for field, model_nn, db_nn in mismatched_notnull]))
    if notnull_sync == 'error':
      raise ValueError(msg)
    elif notnull_sync == 'warn':
      warnings.warn(msg)
    elif notnull_sync == 'fix':
      for field_name, model_notnull, db_notnull in mismatched_notnull:
        try:
          # 尝试使用alter_column
          # 主要适用: 只支持添加 NOT NULL，去除 NOT NULL 需要重建表
          model_field = model_fields[field_name]
          new_field = model_field.clone()
          # 保证NOT NULL和model一致
          new_field.null = not model_notnull
          operations.append(migrator.alter_column_type(
              table_name, field_name, new_field))
        except Exception as e:
          # 如果alter失败，抛异常告知
          raise RuntimeError(
              f"Cannot sync NOT NULL for column '{field_name}'. "
              f"Model: {'NOT NULL' if model_notnull else 'NULLABLE'}, "
              f"DB: {'NOT NULL' if db_notnull else 'NULLABLE'}. "
              f"Reason: {e}\n"
              "On SQLite, removing NOT NULL usually requires manual migration (copy table)."
          )

  # 6. 执行迁移
  if operations:
    migrate(*operations)

  return db
