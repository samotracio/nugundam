from nugundam.angular.models import (
    AngularAutoConfig,
    AngularAutoCountsConfig,
    AngularBinning,
    AngularCrossConfig,
    AngularCrossCountsConfig,
    ConfigDescription,
)


def test_config_describe_class_and_instance():
    desc = AngularAutoConfig.describe()
    assert isinstance(desc, ConfigDescription)
    text = str(desc)
    assert "AngularAutoConfig" in text
    assert "estimator : Literal['NAT', 'DP', 'LS'] = Angular auto-correlation estimator" in text
    assert "binning : AngularBinning = Angular separation binning specification." in text
    assert "nthreads : int = Number of OpenMP threads" in text
    assert repr(desc) == text

    cfg = AngularAutoConfig()
    assert str(cfg.describe()) == text


def test_config_describe_recursive_includes_nested_sections():
    text = str(AngularAutoConfig.describe(recursive=True))
    assert "  AngularBinning" in text
    assert "  nsep : int = Number of angular separation bins." in text
    assert "  ProgressSpec" in text
    assert "  poll_interval : float = Polling interval in seconds for notebook progress watchers." in text


def test_other_config_descriptions_are_available():
    assert "sepmin : float = Minimum angular separation" in str(AngularBinning.describe())
    cross_text = str(AngularCrossConfig.describe())
    assert "columns_data2 : CatalogColumns = Column names for the second data catalog." in cross_text


def test_config_description_has_notebook_friendly_hooks():
    desc = AngularAutoConfig.describe(recursive=True)
    md = desc._repr_markdown_()
    assert md.startswith("```text\nAngularAutoConfig")
    assert md.endswith("\n```")


def test_count_only_config_descriptions_are_available_and_recursive():
    auto_text = str(AngularAutoCountsConfig.describe(recursive=True))
    assert "AngularAutoCountsConfig" in auto_text
    assert "columns : CatalogColumns = Column names for the input catalog used by count-only angular auto-pair runs." in auto_text
    assert "  CatalogColumns" in auto_text
    assert "  ra : str = Right ascension column name." in auto_text

    cross_text = str(AngularCrossCountsConfig.describe(recursive=True))
    assert "AngularCrossCountsConfig" in cross_text
    assert "columns1 : CatalogColumns = Column names for the first input catalog used by count-only angular cross-pair runs." in cross_text
    assert "columns2 : CatalogColumns = Column names for the second input catalog used by count-only angular cross-pair runs." in cross_text
