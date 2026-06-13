# =============================================================================
# Sistema Inteligente de Semáforos y Vigilancia para La Paz
# Paquete: api
# Descripción: Contiene los Blueprints de Flask para exponer los modelos de IA
#              como servicios REST, con persistencia SQLAlchemy y concurrencia
#              mediante ThreadPoolExecutor.
# =============================================================================

from api.semaforo_api import semaforo_bp

__all__ = ["semaforo_bp"]
