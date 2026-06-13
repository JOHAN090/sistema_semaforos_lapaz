/**
 * ============================================================================
 * S.I.S.V. La Paz - Motor de Simulación e Integración API
 * ============================================================================
 * Este script maneja:
 * 1. El renderizado del Canvas (calles, autos).
 * 2. La lógica física 2D de los autos (movimiento, detención).
 * 3. La recolección de métricas simuladas (flujo, densidad).
 * 4. La comunicación con la API Flask (/api/fase_optima) para decidir
 *    los cambios de luz del semáforo.
 */

// ─── CONFIGURACIÓN DEL CANVAS Y ESTADO ─────────────────────────────────────
const canvas = document.getElementById('sim-canvas');
const ctx = canvas.getContext('2d');

let isPaused = false;
let animationId;

// Estado lógico de las 4 fases
// Fases de la IA: 0: NS Verde, 1: EO Verde, 2: Giro, 3: Todo Rojo
let currentPhaseIndex = 3; 
let secondsRemaining = 5; // Empezamos con 5 segundos en rojo

// Estado de luces físicas para [Norte, Sur, Este, Oeste]
// true = Verde, false = Rojo
let physicalLights = {
    N: false,
    S: false,
    E: false,
    O: false
};

// Arreglo de vehículos y peatones activos
let cars = [];
let pedestrians = [];

// Métricas recolectadas (para enviar a la IA)
let metrics = {
    flujo_ns: 0,
    flujo_eo: 0,
    velocidades: [],
};

// ─── DIBUJO DEL ENTORNO (CALLES) ──────────────────────────────────────────
function drawEnvironment() {
    // Fondo oscuro (asfalto)
    ctx.fillStyle = '#1A2234';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const roadWidth = 140;

    ctx.fillStyle = '#2A344A'; // Color calle
    
    // Calle Vertical (Norte-Sur)
    ctx.fillRect(cx - roadWidth/2, 0, roadWidth, canvas.height);
    // Calle Horizontal (Este-Oeste)
    ctx.fillRect(0, cy - roadWidth/2, canvas.width, roadWidth);

    // Líneas separadoras
    ctx.strokeStyle = '#FCD34D'; // Línea amarilla doble centro
    ctx.lineWidth = 4;
    ctx.setLineDash([0, 0]); // Sólida

    // Vertical centro
    ctx.beginPath();
    ctx.moveTo(cx, 0); ctx.lineTo(cx, cy - roadWidth/2);
    ctx.moveTo(cx, cy + roadWidth/2); ctx.lineTo(cx, canvas.height);
    ctx.stroke();

    // Horizontal centro
    ctx.beginPath();
    ctx.moveTo(0, cy); ctx.lineTo(cx - roadWidth/2, cy);
    ctx.moveTo(cx + roadWidth/2, cy); ctx.lineTo(canvas.width, cy);
    ctx.stroke();

    // Líneas blancas punteadas de carril
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 2;
    ctx.setLineDash([15, 15]);
    
    // N-S carriles
    ctx.beginPath();
    ctx.moveTo(cx - roadWidth/4, 0); ctx.lineTo(cx - roadWidth/4, canvas.height);
    ctx.moveTo(cx + roadWidth/4, 0); ctx.lineTo(cx + roadWidth/4, canvas.height);
    ctx.stroke();

    // E-O carriles
    ctx.beginPath();
    ctx.moveTo(0, cy - roadWidth/4); ctx.lineTo(canvas.width, cy - roadWidth/4);
    ctx.moveTo(0, cy + roadWidth/4); ctx.lineTo(canvas.width, cy + roadWidth/4);
    ctx.stroke();

    // Reset dashes
    ctx.setLineDash([]);

    // Dibujar pasos peatonales (cebras)
    ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
    const cw = 20; // crosswalk width
    const stripeW = 4;
    const stripeGap = 6;
    
    function drawZebra(startX, startY, w, h, isVertical) {
        if(isVertical) {
            for(let y = startY; y < startY + h; y += stripeW + stripeGap) {
                ctx.fillRect(startX, y, w, stripeW);
            }
        } else {
            for(let x = startX; x < startX + w; x += stripeW + stripeGap) {
                ctx.fillRect(x, startY, stripeW, h);
            }
        }
    }

    // Norte
    drawZebra(cx - roadWidth/2, cy - roadWidth/2 - cw, roadWidth, cw, false);
    // Sur
    drawZebra(cx - roadWidth/2, cy + roadWidth/2, roadWidth, cw, false);
    // Este
    drawZebra(cx + roadWidth/2, cy - roadWidth/2, cw, roadWidth, true);
    // Oeste
    drawZebra(cx - roadWidth/2 - cw, cy - roadWidth/2, cw, roadWidth, true);
}

// ─── CLASE CAR (VEHÍCULOS) ────────────────────────────────────────────────
class Car {
    constructor() {
        this.width = 18;
        this.length = 36;
        this.speed = Math.random() * 2 + 2; // px per frame
        this.originalSpeed = this.speed;
        
        // Colores realistas
        const colors = ['#EF4444', '#3B82F6', '#FFFFFF', '#10B981', '#9CA3AF', '#F59E0B'];
        this.color = colors[Math.floor(Math.random() * colors.length)];
        
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;
        const rw = 140; // roadWidth

        // Determinar origen y dirección (0: N->S, 1: S->N, 2: W->E, 3: E->W)
        this.direction = Math.floor(Math.random() * 4);
        
        switch(this.direction) {
            case 0: // Viene del Norte hacia el Sur
                this.x = cx - rw/4 - this.width/2; 
                this.y = -this.length;
                this.axis = 'Y';
                this.sign = 1;
                this.lightDir = 'N';
                break;
            case 1: // Viene del Sur hacia el Norte
                this.x = cx + rw/4 - this.width/2; 
                this.y = canvas.height + this.length;
                this.axis = 'Y';
                this.sign = -1;
                this.lightDir = 'S';
                break;
            case 2: // Viene del Oeste hacia el Este
                this.x = -this.length; 
                this.y = cy + rw/4 - this.width/2;
                this.axis = 'X';
                this.sign = 1;
                this.lightDir = 'O';
                break;
            case 3: // Viene del Este hacia el Oeste
                this.x = canvas.width + this.length; 
                this.y = cy - rw/4 - this.width/2;
                this.axis = 'X';
                this.sign = -1;
                this.lightDir = 'E';
                break;
        }

        this.isStopped = false;
    }

    update(allCars) {
        // Lógica de Semáforo (línea de detención)
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;
        const rw = 140;
        const cw = 20; // Ancho cebra
        const stopLineMargin = 5; // Pixeles antes de la cebra

        let stopLineDist = 0;
        let isLightRed = !physicalLights[this.lightDir];

        // Calcular distancia desde el frente del auto hasta la línea de detención
        if (this.direction === 0) stopLineDist = (cy - rw/2 - cw - stopLineMargin) - (this.y + this.length);
        if (this.direction === 1) stopLineDist = this.y - (cy + rw/2 + cw + stopLineMargin);
        if (this.direction === 2) stopLineDist = (cx - rw/2 - cw - stopLineMargin) - (this.x + this.length);
        if (this.direction === 3) stopLineDist = this.x - (cx + rw/2 + cw + stopLineMargin);

        // Lógica para semáforo rojo
        let shouldStopForLight = isLightRed && stopLineDist > 0 && stopLineDist < 40;
        
        if (shouldStopForLight) {
            this.speed = Math.max(0, this.speed - 0.2); // Frenado suave
            if(this.speed < 0.1) {
                this.speed = 0;
                this.isStopped = true;
            }
        } else {
            // Acelerar si está verde
            if (this.speed < this.originalSpeed) {
                this.speed += 0.05;
                this.isStopped = false;
            }
        }

        // Lógica de colisión con el auto de adelante
        let minSafeDist = 15; // Distancia libre entre autos
        let vehicleAhead = null;
        let distToAhead = Infinity;

        for(let other of allCars) {
            if(other !== this && other.direction === this.direction) {
                let d = 0;
                // Calculamos distancia de la parte delantera de ESTE auto a la parte trasera del OTRO
                if(this.direction === 0 && other.y > this.y) d = other.y - (this.y + this.length);
                if(this.direction === 1 && other.y < this.y) d = this.y - (other.y + other.length);
                if(this.direction === 2 && other.x > this.x) d = other.x - (this.x + this.length);
                if(this.direction === 3 && other.x < this.x) d = this.x - (other.x + other.length);
                
                if(d > 0 && d < distToAhead) {
                    distToAhead = d;
                    vehicleAhead = other;
                }
            }
        }

        // Frenar para no chocar con el auto de adelante
        if(vehicleAhead && distToAhead < minSafeDist) {
            this.speed = vehicleAhead.speed;
            this.isStopped = vehicleAhead.isStopped;
        }

        // Mover
        if (this.axis === 'Y') this.y += this.speed * this.sign;
        if (this.axis === 'X') this.x += this.speed * this.sign;

        // Recolectar métricas en tiempo real
        if(this.speed > 0) metrics.velocidades.push(this.speed * 10); // Escalar pseudo km/h
    }

    draw() {
        ctx.fillStyle = this.color;
        // Sombra
        ctx.shadowColor = 'rgba(0,0,0,0.5)';
        ctx.shadowBlur = 5;
        ctx.shadowOffsetY = 2;

        if (this.axis === 'Y') {
            ctx.fillRect(this.x, this.y, this.width, this.length);
            // Luces de freno si está parado
            if (this.speed < 0.5) {
                ctx.fillStyle = '#FF0000';
                ctx.shadowColor = '#FF0000';
                if(this.direction === 0) { // luces arriba
                    ctx.fillRect(this.x + 2, this.y + 2, 4, 3);
                    ctx.fillRect(this.x + this.width - 6, this.y + 2, 4, 3);
                } else { // luces abajo
                    ctx.fillRect(this.x + 2, this.y + this.length - 5, 4, 3);
                    ctx.fillRect(this.x + this.width - 6, this.y + this.length - 5, 4, 3);
                }
            }
        } else {
            ctx.fillRect(this.x, this.y, this.length, this.width);
            if (this.speed < 0.5) {
                ctx.fillStyle = '#FF0000';
                ctx.shadowColor = '#FF0000';
                if(this.direction === 2) { // luces izquierda
                    ctx.fillRect(this.x + 2, this.y + 2, 3, 4);
                    ctx.fillRect(this.x + 2, this.y + this.width - 6, 3, 4);
                } else { // luces derecha
                    ctx.fillRect(this.x + this.length - 5, this.y + 2, 3, 4);
                    ctx.fillRect(this.x + this.length - 5, this.y + this.width - 6, 3, 4);
                }
            }
        }
        ctx.shadowBlur = 0; // reset
    }

    isOutOfBounds() {
        return (this.x < -100 || this.x > canvas.width + 100 || 
                this.y < -100 || this.y > canvas.height + 100);
    }
}

// ─── CLASE PEDESTRIAN (PEATONES) ──────────────────────────────────────────
class Pedestrian {
    constructor() {
        this.radius = 3.5;
        this.speed = Math.random() * 0.8 + 1.2; // Caminan más rápido para no quedar atrapados
        const colors = ['#FBBF24', '#A78BFA', '#F472B6', '#60A5FA', '#FFFFFF'];
        this.color = colors[Math.floor(Math.random() * colors.length)];
        
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;
        const rw = 140;
        const cw = 20;
        
        // 0: Norte, 1: Sur, 2: Este, 3: Oeste
        this.crosswalk = Math.floor(Math.random() * 4);
        this.dir = Math.random() > 0.5 ? 1 : -1; 
        
        // Posicionamiento inicial en las esquinas
        if(this.crosswalk === 0) { // N (Horizontal)
            this.x = this.dir === 1 ? cx - rw/2 - 10 : cx + rw/2 + 10;
            this.y = cy - rw/2 - cw/2 + (Math.random()*12 - 6);
            this.axis = 'X';
        } else if(this.crosswalk === 1) { // S (Horizontal)
            this.x = this.dir === 1 ? cx - rw/2 - 10 : cx + rw/2 + 10;
            this.y = cy + rw/2 + cw/2 + (Math.random()*12 - 6);
            this.axis = 'X';
        } else if(this.crosswalk === 2) { // E (Vertical)
            this.x = cx + rw/2 + cw/2 + (Math.random()*12 - 6);
            this.y = this.dir === 1 ? cy - rw/2 - 10 : cy + rw/2 + 10;
            this.axis = 'Y';
        } else { // O (Vertical)
            this.x = cx - rw/2 - cw/2 + (Math.random()*12 - 6);
            this.y = this.dir === 1 ? cy - rw/2 - 10 : cy + rw/2 + 10;
            this.axis = 'Y';
        }
        
        this.isCrossing = false;
        this.done = false;
    }
    
    update() {
        // ¿Es seguro cruzar?
        let canCross = false;
        if(currentPhaseIndex === 3) canCross = true; // Todo rojo = seguro
        else if(currentPhaseIndex === 0 && (this.crosswalk === 2 || this.crosswalk === 3)) canCross = true; // NS Verde = seguro cruzar E/O
        else if(currentPhaseIndex === 1 && (this.crosswalk === 0 || this.crosswalk === 1)) canCross = true; // EO Verde = seguro cruzar N/S
        
        if(!this.isCrossing && canCross) {
            this.isCrossing = true;
        } else if (!this.isCrossing && !canCross) {
            return; // Esperando en la acera
        }
        
        if (this.axis === 'X') this.x += this.speed * this.dir;
        if (this.axis === 'Y') this.y += this.speed * this.dir;
        
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;
        const rw = 140;
        
        if(this.axis === 'X') {
            if((this.dir === 1 && this.x > cx + rw/2 + 15) || (this.dir === -1 && this.x < cx - rw/2 - 15)) this.done = true;
        } else {
            if((this.dir === 1 && this.y > cy + rw/2 + 15) || (this.dir === -1 && this.y < cy - rw/2 - 15)) this.done = true;
        }
    }
    
    draw() {
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fill();
    }
}

// ─── LOOP PRINCIPAL DE ANIMACIÓN ──────────────────────────────────────────
function animate() {
    if (!isPaused) {
        drawEnvironment();

        // Spawn cars aleatoriamente
        if (Math.random() < 0.03) {
            let newCar = new Car();
            cars.push(newCar);
            // Registrar métrica de flujo
            if(newCar.direction <= 1) metrics.flujo_ns++;
            else metrics.flujo_eo++;
        }

        // Update & Draw cars
        for (let i = cars.length - 1; i >= 0; i--) {
            cars[i].update(cars);
            cars[i].draw();
            
            if (cars[i].isOutOfBounds()) {
                cars.splice(i, 1);
            }
        }

        // Spawn pedestrians (menos cantidad)
        if (Math.random() < 0.01) {
            pedestrians.push(new Pedestrian());
        }

        // Update & Draw pedestrians
        for (let i = pedestrians.length - 1; i >= 0; i--) {
            pedestrians[i].update();
            pedestrians[i].draw();
            if (pedestrians[i].done) {
                pedestrians.splice(i, 1);
            }
        }

        actualizarUI();
    }
    animationId = requestAnimationFrame(animate);
}

// ─── INTEGRACIÓN CON API FLASK (IA) ───────────────────────────────────────

// Actualiza las luces en HTML y estado físico
function setTrafficLights(phaseIndex) {
    // Apagar todas las luces físicas
    physicalLights.N = false; physicalLights.S = false;
    physicalLights.E = false; physicalLights.O = false;
    
    // Apagar CSS classes
    document.querySelectorAll('.light').forEach(l => l.classList.remove('active'));

    // Configurar según fase de la IA
    // 0: N-S Verde, 1: E-O Verde, 2: Giro (simularemos N-S verde tb), 3: Todo Rojo
    if (phaseIndex === 0 || phaseIndex === 2) {
        physicalLights.N = true; physicalLights.S = true;
        document.querySelector('#tl-0 .green').classList.add('active'); // N
        document.querySelector('#tl-1 .green').classList.add('active'); // S
        document.querySelector('#tl-2 .red').classList.add('active'); // E
        document.querySelector('#tl-3 .red').classList.add('active'); // O
    } else if (phaseIndex === 1) {
        physicalLights.E = true; physicalLights.O = true;
        document.querySelector('#tl-0 .red').classList.add('active');
        document.querySelector('#tl-1 .red').classList.add('active');
        document.querySelector('#tl-2 .green').classList.add('active');
        document.querySelector('#tl-3 .green').classList.add('active');
    } else {
        // Todo Rojo (Peatonal)
        document.querySelectorAll('.red').forEach(l => l.classList.add('active'));
    }
}

function generateSimulatedPayload() {
    // Calcular métricas actuales
    let vel_promedio = 0;
    if(metrics.velocidades.length > 0) {
        vel_promedio = metrics.velocidades.reduce((a,b)=>a+b,0) / metrics.velocidades.length;
    }
    
    // Contar colas (autos detenidos) para densidad
    let autosDetenidosNs = cars.filter(c => c.direction <= 1 && c.isStopped).length;
    let autosDetenidosEo = cars.filter(c => c.direction > 1 && c.isStopped).length;

    let payload = {
        "interseccion_id": "INT-001-PRADO",
        "datos_sensores": {
            // Repartir el flujo acumulado en los dos sensores N-S y E-O
            "flujo_ns_1": Math.floor(metrics.flujo_ns / 2) + Math.floor(Math.random()*3), 
            "flujo_ns_2": Math.floor(metrics.flujo_ns / 2) + Math.floor(Math.random()*3),
            "flujo_eo_1": Math.floor(metrics.flujo_eo / 2) + Math.floor(Math.random()*3), 
            "flujo_eo_2": Math.floor(metrics.flujo_eo / 2) + Math.floor(Math.random()*3),
            
            "velocidad_ns_1": vel_promedio > 0 ? vel_promedio : 30.0,
            "velocidad_ns_2": vel_promedio > 0 ? vel_promedio : 30.0,
            "velocidad_eo_1": vel_promedio > 0 ? vel_promedio : 30.0,
            "velocidad_eo_2": vel_promedio > 0 ? vel_promedio : 30.0,
            
            "densidad_ns_1": autosDetenidosNs * 10.0, "densidad_ns_2": autosDetenidosNs * 10.0,
            "densidad_eo_1": autosDetenidosEo * 10.0, "densidad_eo_2": autosDetenidosEo * 10.0,
            
            "hora": new Date().getHours(), "dia_semana": new Date().getDay(), "mes": new Date().getMonth()+1,
            "lluvia_mm": 0.0, "temperatura": 14.0, "visibilidad_km": 10.0,
            
            "peatones_cruce_1": Math.floor(Math.random()*20), "peatones_cruce_2": Math.floor(Math.random()*20),
            "ocupacion_sensor_1": Math.min(1.0, autosDetenidosNs * 0.1), 
            "ocupacion_sensor_2": Math.min(1.0, autosDetenidosNs * 0.1),
            "ocupacion_sensor_3": Math.min(1.0, autosDetenidosEo * 0.1), 
            "ocupacion_sensor_4": Math.min(1.0, autosDetenidosEo * 0.1)
        }
    };

    // Resetear métricas para el siguiente ciclo
    metrics.flujo_ns = 0; metrics.flujo_eo = 0; metrics.velocidades = [];
    
    return payload;
}

async function requestIA_Decision() {
    try {
        const payload = generateSimulatedPayload();
        
        const response = await fetch('/api/fase_optima', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error("Error en servidor IA");
        
        const data = await response.json();
        
        // Aplicar decisión de IA
        currentPhaseIndex = data.fase_recomendada;
        // Para que la simulación no sea aburrida, acortamos los tiempos reales (ej. 40s -> 8s)
        secondsRemaining = Math.max(4, Math.floor(data.tiempo_verde_segundos / 5)); 
        
        // Si la IA decide Fase 3 (Todo Rojo Peatonal), la hacemos muy corta visualmente (3 segundos) 
        // para que el tráfico no se sienta atascado demasiado tiempo en la simulación.
        if (currentPhaseIndex === 3) {
            secondsRemaining = 3;
        }
        
        setTrafficLights(currentPhaseIndex);
        updateAIPanel(data, payload.datos_sensores);

    } catch (error) {
        console.error(error);
        document.getElementById('server-status').innerText = 'Error de Conexión';
        document.getElementById('server-status').className = 'value danger';
    }
}

// ─── ACTUALIZACIÓN DE INTERFAZ ────────────────────────────────────────────

function updateAIPanel(data, sensores) {
    document.getElementById('current-phase').innerText = data.nombre_fase;
    
    // Actualizar Confianza Chart
    const confArc = document.getElementById('confidence-arc');
    const confText = document.getElementById('confidence-text');
    const porcentaje = Math.round(data.confianza * 100);
    
    confArc.setAttribute('stroke-dasharray', `${porcentaje}, 100`);
    confText.textContent = `${porcentaje}%`;
    
    // Color semántico
    if(porcentaje > 80) confArc.style.stroke = 'var(--success)';
    else if(porcentaje > 60) confArc.style.stroke = 'var(--warning)';
    else confArc.style.stroke = 'var(--danger)';

    // Actualizar barras de probabilidad Softmax
    for(let i=0; i<4; i++) {
        document.getElementById(`prob-${i}`).style.width = `${Math.round(data.probabilidades[i] * 100)}%`;
    }

    // Actualizar Sensores
    document.getElementById('val-ns').innerText = sensores.flujo_ns_1 + sensores.flujo_ns_2;
    document.getElementById('val-eo').innerText = sensores.flujo_eo_1 + sensores.flujo_eo_2;
    document.getElementById('val-vel').innerText = `${Math.round(sensores.velocidad_ns_1)} km/h`;
    
    let densidadTotal = sensores.densidad_ns_1 + sensores.densidad_eo_1;
    let textDen = "Baja";
    if(densidadTotal > 40) textDen = "Media";
    if(densidadTotal > 80) textDen = "Alta!";
    document.getElementById('val-den').innerText = textDen;
}

function actualizarUI() {
    document.getElementById('current-time').innerText = `${secondsRemaining} s`;
    document.getElementById('timer-display').innerText = `Siguiente decisión en: ${secondsRemaining}s`;
}

// ─── TEMPORIZADOR DEL CICLO SEMAFÓRICO ────────────────────────────────────
setInterval(() => {
    if(!isPaused) {
        if(secondsRemaining > 0) {
            secondsRemaining--;
        } else {
            // Ciclo terminado, consultar a la IA
            requestIA_Decision();
        }
    }
}, 1000);

// ─── INICIALIZACIÓN ───────────────────────────────────────────────────────
document.getElementById('btn-toggle-sim').addEventListener('click', (e) => {
    isPaused = !isPaused;
    e.target.innerText = isPaused ? "Reanudar Simulación" : "Pausar Simulación";
    e.target.className = isPaused ? "btn btn-primary" : "btn btn-primary";
});

// Arrancar luces rojas y simulación
setTrafficLights(3); 
requestIA_Decision(); // Primer llamado inmediato
animate();
