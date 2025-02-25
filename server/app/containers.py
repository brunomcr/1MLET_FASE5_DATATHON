from dependency_injector import containers, providers

from app.core.config import Settings
from app.services.model_service import ModelService
from app.ml import LightFMPredictor, SparkSessionFactory


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        packages=["app.routers.v1.predictions"]
    )
    
    # Config
    config = providers.Singleton(Settings)

    # Spark
    spark_session_factory = providers.Singleton(
        SparkSessionFactory,
        settings=config
    )

    spark_session = providers.Singleton(
        lambda factory: factory.create_spark_session("lightfm_predictor"),
        factory=spark_session_factory
    )

    # Initialize the predictor
    predictor = providers.Singleton(
        LightFMPredictor,
        spark_session=spark_session,
        model_path=config.provided.MODEL_PATH,
        user_features_path=config.provided.USER_FEATURES_PATH,
        item_features_path=config.provided.ITEM_FEATURES_PATH
    )

    # Services
    model_service = providers.Singleton(
        ModelService,
        predictor=predictor
    )