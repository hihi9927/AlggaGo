from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import json
import random
import numpy as np
from typing import List, Dict, Any
from pydantic import BaseModel
import asyncio
from datetime import datetime

# FastAPI 앱 생성
app = FastAPI(
    title="AlggaGo Game API",
    description="AI 강화학습 기반 바둑돌 게임 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 게임 상태 관리
class GameState:
    def __init__(self):
        self.board_width = 800
        self.board_height = 600
        self.stone_radius = 20
        self.reset_game()
    
    def reset_game(self):
        """게임 초기화"""
        self.black_stones = [
            (200, 100), (400, 100), (600, 100), (300, 150)
        ]
        self.white_stones = [
            (200, 500), (400, 500), (600, 500), (300, 450)
        ]
        self.current_player = "black"
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.game_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def make_ai_move(self):
        """AI가 자동으로 움직임"""
        if self.game_over:
            return {"status": "game_over", "winner": self.winner}
        
        if self.current_player == "black" and self.black_stones:
            stone_idx = random.randint(0, len(self.black_stones) - 1)
            stone_pos = self.black_stones[stone_idx]
            
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(50, 150)
            
            new_x = stone_pos[0] + distance * np.cos(angle)
            new_y = stone_pos[1] + distance * np.sin(angle)
            
            new_x = max(self.stone_radius, min(self.board_width - self.stone_radius, new_x))
            new_y = max(self.stone_radius, min(self.board_height - self.stone_radius, new_y))
            
            self.black_stones[stone_idx] = (new_x, new_y)
            self.check_collisions()
            
            self.current_player = "white"
            self.move_history.append(f"Black moved stone {stone_idx} to ({new_x:.1f}, {new_y:.1f})")
            
            return {
                "status": "success",
                "player": "black",
                "stone_idx": stone_idx,
                "new_position": (new_x, new_y),
                "move_history": self.move_history[-5:]  # 최근 5개만
            }
        
        elif self.current_player == "white" and self.white_stones:
            stone_idx = random.randint(0, len(self.white_stones) - 1)
            stone_pos = self.white_stones[stone_idx]
            
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(50, 150)
            
            new_x = stone_pos[0] + distance * np.cos(angle)
            new_y = stone_pos[1] + distance * np.sin(angle)
            
            new_x = max(self.stone_radius, min(self.board_width - self.stone_radius, new_x))
            new_y = max(self.stone_radius, min(self.board_height - self.stone_radius, new_y))
            
            self.white_stones[stone_idx] = (new_x, new_y)
            self.check_collisions()
            
            self.current_player = "black"
            self.move_history.append(f"White moved stone {stone_idx} to ({new_x:.1f}, {new_y:.1f})")
            
            return {
                "status": "success",
                "player": "white",
                "stone_idx": stone_idx,
                "new_position": (new_x, new_y),
                "move_history": self.move_history[-5:]
            }
        
        return {"status": "no_moves_available"}
    
    def check_collisions(self):
        """돌 간 충돌 체크"""
        collision_threshold = self.stone_radius * 2
        
        for i, black_pos in enumerate(self.black_stones[:]):
            for j, white_pos in enumerate(self.white_stones[:]):
                distance = np.sqrt((black_pos[0] - white_pos[0])**2 + (black_pos[1] - white_pos[1])**2)
                if distance < collision_threshold:
                    if random.random() < 0.5:
                        self.black_stones.pop(i)
                        self.move_history.append(f"Black stone {i} removed by collision")
                    else:
                        self.white_stones.pop(j)
                        self.move_history.append(f"White stone {j} removed by collision")
                    break
        
        # 승리 조건 체크
        if not self.black_stones:
            self.game_over = True
            self.winner = "white"
        elif not self.white_stones:
            self.game_over = True
            self.winner = "black"
    
    def get_game_state(self):
        """현재 게임 상태 반환"""
        return {
            "game_id": self.game_id,
            "black_stones": self.black_stones,
            "white_stones": self.white_stones,
            "current_player": self.current_player,
            "game_over": self.game_over,
            "winner": self.winner,
            "move_history": self.move_history[-10:],  # 최근 10개만
            "board_width": self.board_width,
            "board_height": self.board_height,
            "stone_radius": self.stone_radius
        }

# 전역 게임 인스턴스
game_state = GameState()

# Pydantic 모델들
class MoveRequest(BaseModel):
    player: str
    stone_idx: int
    angle: float
    force: float

class GameStats(BaseModel):
    total_games: int
    black_wins: int
    white_wins: int
    average_moves: float

# WebSocket 연결 관리
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

# API 엔드포인트들

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "🎯 AlggaGo Game API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "game_state": "/api/game/state",
            "new_game": "/api/game/new",
            "ai_move": "/api/game/ai-move",
            "auto_game": "/api/game/auto",
            "stats": "/api/stats",
            "websocket": "/ws"
        }
    }

@app.get("/api/game/state")
async def get_game_state():
    """현재 게임 상태 조회"""
    return game_state.get_game_state()

@app.post("/api/game/new")
async def new_game():
    """새 게임 시작"""
    game_state.reset_game()
    await manager.broadcast(json.dumps({
        "type": "new_game",
        "data": game_state.get_game_state()
    }))
    return {"message": "새 게임이 시작되었습니다!", "game_state": game_state.get_game_state()}

@app.post("/api/game/ai-move")
async def ai_move():
    """AI 턴 실행"""
    result = game_state.make_ai_move()
    
    # WebSocket으로 실시간 업데이트 전송
    await manager.broadcast(json.dumps({
        "type": "ai_move",
        "data": {
            "result": result,
            "game_state": game_state.get_game_state()
        }
    }))
    
    return result

@app.post("/api/game/auto")
async def auto_game(turns: int = 10):
    """자동 게임 실행"""
    results = []
    for i in range(turns):
        if game_state.game_over:
            break
        result = game_state.make_ai_move()
        results.append(result)
        await asyncio.sleep(0.1)  # 약간의 딜레이
    
    await manager.broadcast(json.dumps({
        "type": "auto_game_complete",
        "data": {
            "results": results,
            "game_state": game_state.get_game_state()
        }
    }))
    
    return {
        "message": f"자동 게임 완료 ({len(results)}턴)",
        "results": results,
        "final_state": game_state.get_game_state()
    }

@app.get("/api/stats")
async def get_stats():
    """게임 통계 조회"""
    # 샘플 통계 데이터
    stats = {
        "total_games": 150,
        "black_wins": 78,
        "white_wins": 72,
        "average_moves": 12.5,
        "current_game": {
            "moves": len(game_state.move_history),
            "black_stones": len(game_state.black_stones),
            "white_stones": len(game_state.white_stones)
        },
        "performance": {
            "regular_attack_success": 0.75,
            "split_attack_success": 0.45,
            "overall_win_rate": 0.65
        }
    }
    return stats

@app.get("/api/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# WebSocket 엔드포인트
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "get_state":
                await manager.send_personal_message(
                    json.dumps({
                        "type": "game_state",
                        "data": game_state.get_game_state()
                    }),
                    websocket
                )
            elif message["type"] == "ai_move":
                result = game_state.make_ai_move()
                await manager.broadcast(json.dumps({
                    "type": "ai_move",
                    "data": {
                        "result": result,
                        "game_state": game_state.get_game_state()
                    }
                }))
            elif message["type"] == "new_game":
                game_state.reset_game()
                await manager.broadcast(json.dumps({
                    "type": "new_game",
                    "data": game_state.get_game_state()
                }))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# 정적 파일 서빙 (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/game", response_class=HTMLResponse)
async def game_page():
    """게임 웹 페이지"""
    html_content = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🎯 AlggaGo - AI 바둑돌 게임</title>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            .header h1 {
                color: #333;
                font-size: 2.5rem;
                margin: 0;
            }
            .game-board {
                width: 800px;
                height: 600px;
                border: 3px solid #333;
                border-radius: 10px;
                background-color: #8B4513;
                margin: 20px auto;
                position: relative;
            }
            .stone {
                position: absolute;
                border-radius: 50%;
                border: 2px solid #000;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                font-size: 14px;
                cursor: pointer;
                transition: transform 0.2s;
            }
            .stone:hover {
                transform: scale(1.1);
            }
            .stone.black {
                background-color: #000;
                color: white;
            }
            .stone.white {
                background-color: #fff;
                color: black;
            }
            .controls {
                text-align: center;
                margin: 20px 0;
            }
            .btn {
                background: #4CAF50;
                color: white;
                border: none;
                padding: 12px 24px;
                margin: 5px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                transition: background 0.3s;
            }
            .btn:hover {
                background: #45a049;
            }
            .btn:disabled {
                background: #cccccc;
                cursor: not-allowed;
            }
            .stats {
                display: flex;
                justify-content: space-around;
                margin: 20px 0;
            }
            .stat-card {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                min-width: 120px;
            }
            .stat-value {
                font-size: 2rem;
                font-weight: bold;
                color: #333;
            }
            .stat-label {
                color: #666;
                font-size: 0.9rem;
            }
            .history {
                max-height: 200px;
                overflow-y: auto;
                background: #f8f9fa;
                padding: 15px;
                border-radius: 10px;
                margin: 20px 0;
            }
            .history-item {
                padding: 5px 0;
                border-bottom: 1px solid #ddd;
            }
            .status {
                text-align: center;
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
                font-weight: bold;
            }
            .status.info {
                background: #d1ecf1;
                color: #0c5460;
            }
            .status.success {
                background: #d4edda;
                color: #155724;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 AlggaGo - AI 바둑돌 게임</h1>
                <p>AI 강화학습 기반 전략 게임</p>
            </div>
            
            <div class="controls">
                <button class="btn" onclick="newGame()">새 게임</button>
                <button class="btn" onclick="aiMove()">AI 턴</button>
                <button class="btn" onclick="autoGame()">자동 게임 (10턴)</button>
                <button class="btn" onclick="resetGame()">리셋</button>
            </div>
            
            <div id="status" class="status info">게임 준비 중...</div>
            
            <div class="game-board" id="gameBoard">
                <!-- 돌들이 여기에 동적으로 추가됩니다 -->
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value" id="blackCount">4</div>
                    <div class="stat-label">검은 돌</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="whiteCount">4</div>
                    <div class="stat-label">흰 돌</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="currentPlayer">검은</div>
                    <div class="stat-label">현재 턴</div>
                </div>
            </div>
            
            <div class="history">
                <h3>게임 히스토리</h3>
                <div id="historyList">
                    <div class="history-item">게임이 시작되었습니다.</div>
                </div>
            </div>
        </div>

        <script>
            let gameState = {
                black_stones: [],
                white_stones: [],
                current_player: 'black',
                game_over: false,
                winner: null,
                move_history: []
            };

            // WebSocket 연결
            const ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onopen = function() {
                console.log('WebSocket 연결됨');
                updateStatus('게임 준비 완료', 'info');
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'game_state') {
                    updateGameState(data.data);
                } else if (data.type === 'ai_move') {
                    updateGameState(data.data.game_state);
                    updateStatus(`${data.data.result.player} 턴 완료!`, 'success');
                } else if (data.type === 'new_game') {
                    updateGameState(data.data);
                    updateStatus('새 게임이 시작되었습니다!', 'success');
                }
            };

            async function newGame() {
                try {
                    const response = await fetch('/api/game/new', { method: 'POST' });
                    const data = await response.json();
                    updateGameState(data.game_state);
                    updateStatus('새 게임이 시작되었습니다!', 'success');
                } catch (error) {
                    console.error('Error:', error);
                    updateStatus('게임 시작 중 오류가 발생했습니다.', 'error');
                }
            }

            async function aiMove() {
                try {
                    const response = await fetch('/api/game/ai-move', { method: 'POST' });
                    const data = await response.json();
                    if (data.status === 'success') {
                        updateStatus(`${data.player} 턴 완료!`, 'success');
                    }
                } catch (error) {
                    console.error('Error:', error);
                    updateStatus('AI 턴 실행 중 오류가 발생했습니다.', 'error');
                }
            }

            async function autoGame() {
                try {
                    updateStatus('자동 게임 진행 중...', 'info');
                    const response = await fetch('/api/game/auto?turns=10', { method: 'POST' });
                    const data = await response.json();
                    updateGameState(data.final_state);
                    updateStatus('자동 게임이 완료되었습니다!', 'success');
                } catch (error) {
                    console.error('Error:', error);
                    updateStatus('자동 게임 중 오류가 발생했습니다.', 'error');
                }
            }

            function resetGame() {
                newGame();
            }

            function updateGameState(state) {
                gameState = state;
                renderGame();
                updateStats();
                updateHistory();
            }

            function renderGame() {
                const board = document.getElementById('gameBoard');
                board.innerHTML = '';
                
                // 검은 돌들
                gameState.black_stones.forEach((pos, idx) => {
                    const stone = document.createElement('div');
                    stone.className = 'stone black';
                    stone.style.left = (pos[0] - 20) + 'px';
                    stone.style.top = (pos[1] - 20) + 'px';
                    stone.style.width = '40px';
                    stone.style.height = '40px';
                    stone.textContent = idx + 1;
                    board.appendChild(stone);
                });
                
                // 흰 돌들
                gameState.white_stones.forEach((pos, idx) => {
                    const stone = document.createElement('div');
                    stone.className = 'stone white';
                    stone.style.left = (pos[0] - 20) + 'px';
                    stone.style.top = (pos[1] - 20) + 'px';
                    stone.style.width = '40px';
                    stone.style.height = '40px';
                    stone.textContent = idx + 1;
                    board.appendChild(stone);
                });
            }

            function updateStats() {
                document.getElementById('blackCount').textContent = gameState.black_stones.length;
                document.getElementById('whiteCount').textContent = gameState.white_stones.length;
                document.getElementById('currentPlayer').textContent = 
                    gameState.current_player === 'black' ? '검은' : '흰';
            }

            function updateHistory() {
                const historyList = document.getElementById('historyList');
                historyList.innerHTML = '';
                
                gameState.move_history.slice(-10).forEach(move => {
                    const item = document.createElement('div');
                    item.className = 'history-item';
                    item.textContent = move;
                    historyList.appendChild(item);
                });
            }

            function updateStatus(message, type) {
                const status = document.getElementById('status');
                status.textContent = message;
                status.className = `status ${type}`;
            }

            // 초기 게임 상태 로드
            window.onload = async function() {
                try {
                    const response = await fetch('/api/game/state');
                    const data = await response.json();
                    updateGameState(data);
                } catch (error) {
                    console.error('Error loading game state:', error);
                }
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
