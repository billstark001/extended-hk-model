from typing import Dict, Type, Literal, Any, Optional, Union
import io
import peewee
from playhouse.migrate import SqliteMigrator, migrate
import numpy as np


class NumpyArrayField(peewee.BlobField):
  """
  支持自动序列化/反序列化 numpy.ndarray 的 Peewee 字段。
  支持 list 自动转 np.ndarray。
  支持可选强制类型（force_dtype），在读写时用 astype。
  """

  def __init__(self, force_dtype: Optional[Union[str, np.dtype]] = None, *args, **kwargs):
    """
    :param force_dtype: 强制类型（如np.float32），None则不转换。
    其余参数同peewee.BlobField
    """
    self.force_dtype = np.dtype(
        force_dtype) if force_dtype is not None else None
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
    extra_columns: Literal['ignore', 'error', 'delete'] = 'ignore'
) -> peewee.Database:
  db_is_none = db is None
  if db_is_none:
    db = model._meta.database
  table_name: str = model._meta.table_name

  # 1. 获取模型字段名
  model_fields: Dict[str, Any] = model._meta.fields.copy()
  model_field_names = set(model_fields.keys())

  # 2. 获取数据库字段名
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

  # 5. 执行迁移
  if operations:
    migrate(*operations)

  return db


def sync_peewee_table_naive(
    db: Optional[peewee.Database],
    model: Type[peewee.Model],
    extra_columns: Literal['ignore', 'error', 'delete'] = 'ignore'
) -> peewee.Database:
  """
  同步 peewee 模型与数据库表结构。

  :param model: peewee Model 子类
  :param extra_columns: 多余列的处理方式
  """
  db_is_none = db is None
  if db_is_none:
    db = model._meta.database
  table_name: str = model._meta.table_name
  db.connect(reuse_if_open=True)

  # 1. 冷启动：表不存在则直接创建
  if not db.table_exists(table_name):
    model.create_table()
    print(f"Table {table_name} created (cold start).")
    return

  # 2. 获取模型字段和数据库字段
  model_fields: Dict[str, Any] = model._meta.fields
  db_fields = {row.name: row for row in db.get_columns(table_name)}

  model_field_names = set(model_fields.keys())
  db_field_names = set(db_fields.keys())

  # 3. 多余字段
  extra_in_db = db_field_names - model_field_names
  if extra_in_db:
    if extra_columns == 'error':
      raise ValueError(f"Table {table_name} has extra columns: {extra_in_db}")
    elif extra_columns == 'delete':
      # 重建表并迁移数据
      tmp_table = f"{table_name}_tmp_sync"
      model.create_table(table_name=tmp_table)
      # 只迁移交集字段
      common_cols = list(model_field_names & db_field_names)
      if 'id' in model_field_names:
        # id字段优先
        common_cols = ['id'] + [c for c in common_cols if c != 'id']
      col_csv = ', '.join(common_cols)
      db.execute_sql(
          f'INSERT INTO "{tmp_table}" ({col_csv}) SELECT {col_csv} FROM "{table_name}"'
      )
      db.execute_sql(f'DROP TABLE "{table_name}"')
      db.execute_sql(f'ALTER TABLE "{tmp_table}" RENAME TO "{table_name}"')
      print(
          f"Deleted extra columns {extra_in_db} from {table_name} by recreating table.")
      db_fields = {row.name: row for row in db.get_columns(table_name)}
      db_field_names = set(db_fields.keys())

  # 4. 新增模型中有但表里没有的字段
  missing_in_db = model_field_names - db_field_names
  for field_name in missing_in_db:
    field = model_fields[field_name]
    db.add_column(table_name, field_name, field)
    print(f"Added column: {field_name} ({field.get_column_type()})")

  db.commit()

  return db
