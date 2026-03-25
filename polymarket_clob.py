"""
polymarket_clob.py
==================
Camada de acesso ao Polymarket CLOB.

Responsabilidades:
  - Derivar credenciais L2 (API key / secret / passphrase) a partir da private key Polygon
  - Consultar order book (bid, ask, spread, depth)
  - Enriquecer brackets do mercado com dados CLOB em tempo real
  - Colocar ordens limitadas (modo REAL) ou simular (modo PAPER)

Dependência:
    pip install py-clob-client

Variáveis de ambiente:
    POLY_PRIVATE_KEY=0x...    (obrigatória para REAL e para order book autenticado)
    POLY_FUNDER=0x...         (opcional — endereço da wallet funder em contas delegadas)

Uso típico:
    from polymarket_clob import ClobClient, TradingMode, PositionManager, Position
    clob = ClobClient(private_key=..., mode=TradingMode.PAPER, max_daily_loss=50.0)
    book = clob.get_orderbook(token_id)
    result = clob.buy_yes(token_id, price_ask, size_usdc)
    # Posições registadas automaticamente em clob.positions
    clob.positions.refresh(clob)          # actualiza P&L e verifica resolução
    opens = clob.positions.open_positions()
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Polymarket CLOB endpoints ─────────────────────────
CLOB_HOST   = "https://clob.polymarket.com"
CHAIN_ID    = 137   # Polygon mainnet

# ── Constantes de ordem ───────────────────────────────
TICK_SIZE   = 0.01          # mínimo incremento de preço no CLOB
MIN_SIZE    = 5.0           # tamanho mínimo de ordem em USDC (Polymarket exige ≥ $1, usamos $5 por segurança)
FEE_RATE    = 0.0           # Polymarket não cobra fee no CLOB actualmente


# ══════════════════════════════════════════════════════
#  ENUMS / DATACLASSES
# ══════════════════════════════════════════════════════

class TradingMode(Enum):
    PAPER = "paper"
    REAL  = "real"


@dataclass
class OrderBookLevel:
    price: float
    size:  float


@dataclass
class OrderBook:
    token_id:  str
    timestamp: float
    bids:      list[OrderBookLevel] = field(default_factory=list)   # melhor primeiro (mais alto)
    asks:      list[OrderBookLevel] = field(default_factory=list)   # melhor primeiro (mais baixo)

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> float | None:
        if self.best_bid and self.best_ask:
            return round((self.best_bid + self.best_ask) / 2, 4)
        return self.best_ask or self.best_bid

    @property
    def spread(self) -> float | None:
        if self.best_bid and self.best_ask:
            return round(self.best_ask - self.best_bid, 4)
        return None

    @property
    def bid_depth_usdc(self) -> float:
        """USDC total nos top-5 bids."""
        return round(sum(l.price * l.size for l in self.bids[:5]), 2)

    @property
    def ask_depth_usdc(self) -> float:
        """USDC total nos top-5 asks."""
        return round(sum(l.price * l.size for l in self.asks[:5]), 2)


@dataclass
class OrderResult:
    success:    bool
    mode:       TradingMode
    order_id:   str | None     = None
    token_id:   str | None     = None
    side:       str            = "BUY"
    outcome:    str            = "YES"
    price:      float          = 0.0
    size_usdc:  float          = 0.0
    shares:     float          = 0.0
    status:     str            = ""
    error:      str | None     = None
    timestamp:  str            = ""
    simulated:  bool           = False

    def to_dict(self) -> dict:
        return {
            "success":   self.success,
            "mode":      self.mode.value,
            "order_id":  self.order_id,
            "token_id":  self.token_id,
            "side":      self.side,
            "outcome":   self.outcome,
            "price":     self.price,
            "size_usdc": self.size_usdc,
            "shares":    self.shares,
            "status":    self.status,
            "error":     self.error,
            "timestamp": self.timestamp,
            "simulated": self.simulated,
        }


# ══════════════════════════════════════════════════════
#  CLOB CLIENT
# ══════════════════════════════════════════════════════

class ClobClient:
    """
    Cliente CLOB para o Polymarket.

    Parâmetros
    ----------
    private_key : str
        Private key da wallet Polygon (0x...).
    mode : TradingMode
        PAPER → simula ordens, não envia nada.
        REAL  → envia ordens reais ao CLOB.
    max_daily_loss : float
        Stop-loss diário em USDC. Se a perda realizada + ordens abertas
        atingirem este valor, novas ordens são bloqueadas.
    log_dir : Path
        Directório onde guardar logs de ordens.
    """

    def __init__(
        self,
        private_key: str,
        mode: TradingMode = TradingMode.PAPER,
        max_daily_loss: float = 50.0,
        log_dir: Path = Path("live_bot_logs"),
    ):
        if not private_key:
            raise ValueError(
                "POLY_PRIVATE_KEY não definida.\n"
                "  export POLY_PRIVATE_KEY=0x...   (Linux/macOS)\n"
                "  set POLY_PRIVATE_KEY=0x...       (Windows CMD)"
            )

        self.mode           = mode
        self.max_daily_loss = max_daily_loss
        self.log_dir        = log_dir
        self.log_dir.mkdir(exist_ok=True)

        # P&L diário (loss = positivo → perda)
        self._daily_loss:   float      = 0.0
        self._daily_date:   date | None = None
        self._order_log:    list[dict] = []

        # Portfolio de posições
        self.positions: PositionManager = PositionManager(mode, log_dir)

        # Inicializar cliente py-clob-client
        self._client = self._init_clob_client(private_key)

    # ── Inicialização ──────────────────────────────────

    def _init_clob_client(self, private_key: str):
        """
        Inicializa py-clob-client e obtém credenciais L2 via
        create_or_derive_api_creds() — método recomendado que cria
        novas creds se não existirem, ou reutiliza as existentes.

        Suporta POLY_FUNDER para contas com funder separado (carteiras delegadas).
        """
        try:
            from py_clob_client.client import ClobClient as _ClobClient
        except ImportError:
            raise ImportError(
                "py-clob-client não instalado.\n"
                "  pip install py-clob-client"
            )

        # Ler FUNDER da variável de ambiente (opcional)
        funder = os.environ.get("POLY_FUNDER") or None

        # Criar cliente com private key (L1) e funder opcional
        client = _ClobClient(
            host     = CLOB_HOST,
            chain_id = CHAIN_ID,
            key      = private_key,
            **({"funder": funder} if funder else {}),
        )

        # Obter credenciais L2 via create_or_derive_api_creds()
        # Cria novas creds se não existirem; reutiliza se já existirem no servidor.
        try:
            creds = client.create_or_derive_api_creds()
            client.set_api_creds(creds)
            logger.info("Credenciais CLOB obtidas via create_or_derive_api_creds()")
        except Exception as e:
            logger.warning(
                "Falha em create_or_derive_api_creds: %s — order book público ainda funciona", e
            )

        return client

    # ── Order Book ─────────────────────────────────────

    def get_orderbook(self, token_id: str) -> OrderBook | None:
        """
        Consulta o order book do CLOB para um token específico.
        Devolve OrderBook com bids/asks ordenados (melhor primeiro).
        Devolve None se falhar.
        """
        if not token_id:
            return None
        try:
            book_raw = self._client.get_order_book(token_id)
            bids = sorted(
                [OrderBookLevel(float(b.price), float(b.size))
                 for b in (book_raw.bids or [])],
                key=lambda x: -x.price,   # melhor bid = mais alto
            )
            asks = sorted(
                [OrderBookLevel(float(a.price), float(a.size))
                 for a in (book_raw.asks or [])],
                key=lambda x: x.price,    # melhor ask = mais baixo
            )
            return OrderBook(
                token_id  = token_id,
                timestamp = time.time(),
                bids      = bids,
                asks      = asks,
            )
        except Exception as e:
            # 404 = mercado fechado ou token inactivo — esperado, não é erro
            err_str = str(e)
            if "404" not in err_str:
                logger.warning("get_orderbook falhou para %s: %s", token_id, e)
            return None

    def enrich_bracket(self, bracket: dict) -> dict:
        """
        Adiciona dados CLOB (bid, ask, spread) a um bracket do mercado.
        Se o CLOB falhar, mantém o preço Gamma como fallback.
        Returns novo dict enriquecido (não modifica o original).
        """
        b = dict(bracket)
        token_id = b.get("token_id")
        if not token_id:
            b["ask"]    = b.get("price")
            b["bid"]    = b.get("price")
            b["spread"] = None
            b["book"]   = None
            return b

        book = self.get_orderbook(token_id)
        if book and book.best_ask is not None:
            b["ask"]    = book.best_ask
            b["bid"]    = book.best_bid
            b["spread"] = book.spread
            b["mid"]    = book.mid
            b["book"]   = book
            # Actualizar price para o ask (o que pagas ao comprar YES)
            b["price"]  = book.best_ask
        else:
            b["ask"]    = b.get("price")
            b["bid"]    = b.get("price")
            b["spread"] = None
            b["mid"]    = b.get("price")
            b["book"]   = None

        return b

    # ── Gestão de risco ────────────────────────────────

    def _reset_daily_if_needed(self) -> None:
        today = date.today()
        if self._daily_date != today:
            self._daily_date = today
            self._daily_loss = 0.0

    def daily_loss(self) -> float:
        self._reset_daily_if_needed()
        return self._daily_loss

    def stop_loss_triggered(self) -> bool:
        return self.daily_loss() >= self.max_daily_loss

    def record_loss(self, amount: float) -> None:
        """Registar perda realizada (chamado quando uma posição resolve contra nós)."""
        self._reset_daily_if_needed()
        self._daily_loss = round(self._daily_loss + amount, 4)

    # ── Colocação de ordens ────────────────────────────

    def buy_yes(
        self,
        token_id:      str,
        price:         float,
        size_usdc:     float,
        bracket_label: str  = "",
        market_slug:   str  = "",
    ) -> OrderResult:
        """
        Comprar YES num bracket.

        Parâmetros
        ----------
        token_id    : CLOB token ID do outcome YES
        price       : preço limite em USDC por share (0–1), tipicamente o ask
        size_usdc   : montante em USDC a gastar (não shares)
        bracket_label : label do bracket para logging

        Devolve OrderResult com resultado da operação.
        """
        from datetime import datetime as _dt

        self._reset_daily_if_needed()
        ts = _dt.now().isoformat()

        # Validações
        if self.stop_loss_triggered():
            return OrderResult(
                success=False, mode=self.mode,
                error=f"Stop-loss diário atingido (perda={self._daily_loss:.2f} >= max={self.max_daily_loss:.2f})",
                timestamp=ts,
            )

        if size_usdc < MIN_SIZE:
            return OrderResult(
                success=False, mode=self.mode,
                error=f"Tamanho ${size_usdc:.2f} abaixo do mínimo ${MIN_SIZE:.2f}",
                timestamp=ts,
            )

        if not (TICK_SIZE <= price <= 1 - TICK_SIZE):
            return OrderResult(
                success=False, mode=self.mode,
                error=f"Preço {price:.4f} fora do intervalo válido [{TICK_SIZE}, {1-TICK_SIZE}]",
                timestamp=ts,
            )

        shares = round(size_usdc / price, 4)

        # ── PAPER MODE ────────────────────────────────
        if self.mode == TradingMode.PAPER:
            result = OrderResult(
                success    = True,
                mode       = TradingMode.PAPER,
                order_id   = f"PAPER-{int(time.time())}",
                token_id   = token_id,
                side       = "BUY",
                outcome    = "YES",
                price      = round(price, 4),
                size_usdc  = round(size_usdc, 2),
                shares     = shares,
                status     = "SIMULATED",
                timestamp  = ts,
                simulated  = True,
            )
            self._log_order(result, bracket_label)
            # Registar posição no portfolio
            self.positions.add(Position(
                date_opened   = ts[:10],
                bracket_label = bracket_label,
                token_id      = token_id,
                entry_ask     = round(price, 4),
                shares        = shares,
                size_usdc     = round(size_usdc, 2),
                mode          = "paper",
                order_id      = result.order_id,
                market_slug   = market_slug,
            ))
            return result

        # ── REAL MODE ─────────────────────────────────
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType

            # ⚠️  Usa string "BUY"/"SELL" diretamente em vez de importar
            #     a constante BUY do order_builder — mais robusto entre versões
            order_args = OrderArgs(
                token_id = token_id,
                price    = round(price, 4),
                size     = shares,
                side     = "BUY",
            )
            signed_order = self._client.create_order(order_args)
            response     = self._client.post_order(signed_order, OrderType.GTC)

            order_id = response.get("orderID") or response.get("id") or "?"
            status   = response.get("status", "unknown")

            result = OrderResult(
                success    = status in ("matched", "live", "delayed"),
                mode       = TradingMode.REAL,
                order_id   = order_id,
                token_id   = token_id,
                side       = "BUY",
                outcome    = "YES",
                price      = round(price, 4),
                size_usdc  = round(size_usdc, 2),
                shares     = shares,
                status     = status,
                timestamp  = ts,
                simulated  = False,
            )

            if not result.success:
                result.error = f"Status inesperado: {status} | response: {response}"

        except Exception as e:
            result = OrderResult(
                success   = False,
                mode      = TradingMode.REAL,
                token_id  = token_id,
                error     = str(e),
                timestamp = ts,
                simulated = False,
            )
            logger.error("Falha ao colocar ordem REAL: %s", e)

        self._log_order(result, bracket_label)
        # Registar posição no portfolio se a ordem foi aceite
        if result.success:
            self.positions.add(Position(
                date_opened   = ts[:10],
                bracket_label = bracket_label,
                token_id      = token_id,
                entry_ask     = round(price, 4),
                shares        = shares,
                size_usdc     = round(size_usdc, 2),
                mode          = "real",
                order_id      = result.order_id or "",
                market_slug   = market_slug,
            ))
        return result

    # ── Venda / Fecho manual de posição ───────────────

    def sell_yes(
        self,
        position: "Position",
        bid_price: float,
    ) -> "OrderResult":
        """
        Vende (fecha) uma posição YES existente ao bid actual.

        Parâmetros
        ----------
        position  : Position aberta a fechar
        bid_price : melhor bid do CLOB no momento (preço de venda)

        Em PAPER: simula a venda e actualiza o P&L final.
        Em REAL : envia ordem de venda ao CLOB.
        """
        from datetime import datetime as _dt
        ts = _dt.now().isoformat()

        if position.status != PositionStatus.OPEN:
            return OrderResult(
                success=False, mode=self.mode,
                error=f"Posição já fechada (status={position.status.value})",
                timestamp=ts,
            )

        if not (TICK_SIZE <= bid_price <= 1 - TICK_SIZE):
            return OrderResult(
                success=False, mode=self.mode,
                error=f"Bid {bid_price:.4f} fora do intervalo válido",
                timestamp=ts,
            )

        pnl_usd = round((bid_price - position.entry_ask) * position.shares, 2)
        pnl_pct = round((bid_price / position.entry_ask - 1) * 100, 2) if position.entry_ask else 0.0

        # ── PAPER MODE ────────────────────────────────
        if self.mode == TradingMode.PAPER:
            position.status      = PositionStatus.WON if pnl_usd >= 0 else PositionStatus.LOST
            position.pnl_usd     = pnl_usd
            position.pnl_pct     = pnl_pct
            position.current_mid = bid_price
            position.last_updated = ts
            self.positions._save()

            result = OrderResult(
                success   = True,
                mode      = TradingMode.PAPER,
                order_id  = f"PAPER-SELL-{int(time.time())}",
                token_id  = position.token_id,
                side      = "SELL",
                outcome   = "YES",
                price     = round(bid_price, 4),
                size_usdc = round(bid_price * position.shares, 2),
                shares    = position.shares,
                status    = "SIMULATED_SELL",
                timestamp = ts,
                simulated = True,
            )
            self._log_order(result, position.bracket_label)
            return result

        # ── REAL MODE ─────────────────────────────────
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType

            # ⚠️  Usa string "SELL" diretamente — mais robusto entre versões
            order_args = OrderArgs(
                token_id = position.token_id,
                price    = round(bid_price, 4),
                size     = position.shares,
                side     = "SELL",
            )
            signed_order = self._client.create_order(order_args)
            response     = self._client.post_order(signed_order, OrderType.GTC)

            order_id = response.get("orderID") or response.get("id") or "?"
            status   = response.get("status", "unknown")
            success  = status in ("matched", "live", "delayed")

            if success:
                position.status       = PositionStatus.WON if pnl_usd >= 0 else PositionStatus.LOST
                position.pnl_usd      = pnl_usd
                position.pnl_pct      = pnl_pct
                position.current_mid  = bid_price
                position.last_updated = ts
                self.positions._save()

            result = OrderResult(
                success   = success,
                mode      = TradingMode.REAL,
                order_id  = order_id,
                token_id  = position.token_id,
                side      = "SELL",
                outcome   = "YES",
                price     = round(bid_price, 4),
                size_usdc = round(bid_price * position.shares, 2),
                shares    = position.shares,
                status    = status,
                timestamp = ts,
                simulated = False,
                error     = None if success else f"Status inesperado: {status}",
            )

        except Exception as e:
            result = OrderResult(
                success=False, mode=TradingMode.REAL,
                token_id=position.token_id, error=str(e),
                timestamp=ts, simulated=False,
            )
            logger.error("Falha ao fechar posição REAL: %s", e)

        self._log_order(result, position.bracket_label)
        return result

    # ── Saldo ──────────────────────────────────────────

    def get_usdc_balance(self) -> float | None:
        """
        Saldo USDC disponível na wallet do Polymarket.
        Tenta vários métodos pois o nome varia entre versões do py-clob-client.
        """
        for method_name in ("get_collateral_balance", "get_balance",
                            "get_usdc_balance", "get_available_balance"):
            method = getattr(self._client, method_name, None)
            if method is None:
                continue
            try:
                result = method()
                if isinstance(result, (int, float)):
                    return float(result)
                if isinstance(result, str):
                    return float(result)
                if isinstance(result, dict):
                    for key in ("balance", "usdc", "collateral", "available"):
                        if key in result:
                            return float(result[key])
            except Exception as e:
                logger.debug("get_usdc_balance via %s falhou: %s", method_name, e)
                continue

        logger.warning("Nenhum método de saldo disponível no py-clob-client instalado")
        return None

    # ── Logging ────────────────────────────────────────

    def _log_order(self, result: OrderResult, bracket_label: str = "") -> None:
        entry = result.to_dict()
        entry["bracket_label"] = bracket_label
        self._order_log.append(entry)

        log_path = self.log_dir / f"orders_{date.today()}.json"
        try:
            existing = json.loads(log_path.read_text()) if log_path.exists() else []
            existing.append(entry)
            log_path.write_text(json.dumps(existing, indent=2))
        except Exception as e:
            logger.warning("Falha ao guardar log de ordem: %s", e)

    def order_log(self) -> list[dict]:
        return list(self._order_log)


# ══════════════════════════════════════════════════════
#  POSIÇÕES — gestão de portfolio
# ══════════════════════════════════════════════════════

GAMMA_API = "https://gamma-api.polymarket.com"

class PositionStatus(Enum):
    OPEN     = "open"
    WON      = "won"
    LOST     = "lost"
    EXPIRED  = "expired"
    UNKNOWN  = "unknown"


@dataclass
class Position:
    """
    Representa uma posição aberta ou fechada no Polymarket.

    Campos gravados no momento da compra (imutáveis):
        date_opened, bracket_label, token_id, entry_ask,
        shares, size_usdc, mode, order_id, market_slug

    Campos actualizados em tempo real:
        current_mid, status, pnl_usd, pnl_pct, last_updated
    """
    date_opened:    str
    bracket_label:  str
    token_id:       str
    entry_ask:      float
    shares:         float
    size_usdc:      float
    mode:           str
    order_id:       str
    market_slug:    str = ""

    current_mid:    float | None = None
    status:         PositionStatus = PositionStatus.OPEN
    pnl_usd:        float | None = None
    pnl_pct:        float | None = None
    last_updated:   str = ""

    def to_dict(self) -> dict:
        return {
            "date_opened":   self.date_opened,
            "bracket_label": self.bracket_label,
            "token_id":      self.token_id,
            "entry_ask":     self.entry_ask,
            "shares":        self.shares,
            "size_usdc":     self.size_usdc,
            "mode":          self.mode,
            "order_id":      self.order_id,
            "market_slug":   self.market_slug,
            "current_mid":   self.current_mid,
            "status":        self.status.value,
            "pnl_usd":       self.pnl_usd,
            "pnl_pct":       self.pnl_pct,
            "last_updated":  self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        p = cls(
            date_opened   = d["date_opened"],
            bracket_label = d["bracket_label"],
            token_id      = d.get("token_id", ""),
            entry_ask     = float(d["entry_ask"]),
            shares        = float(d["shares"]),
            size_usdc     = float(d["size_usdc"]),
            mode          = d.get("mode", "paper"),
            order_id      = d.get("order_id", ""),
            market_slug   = d.get("market_slug", ""),
            current_mid   = d.get("current_mid"),
            last_updated  = d.get("last_updated", ""),
        )
        try:
            p.status = PositionStatus(d.get("status", "open"))
        except ValueError:
            p.status = PositionStatus.UNKNOWN
        p.pnl_usd = d.get("pnl_usd")
        p.pnl_pct = d.get("pnl_pct")
        return p


class PositionManager:
    """
    Gere o portfolio de posições abertas/fechadas.

    PAPER: persiste tudo em  live_bot_logs/paper_positions.json
    REAL:  persiste tudo em  live_bot_logs/real_positions.json
           + tenta confirmar resolução via Gamma API
    """

    def __init__(self, mode: TradingMode, log_dir: Path = Path("live_bot_logs")):
        self.mode    = mode
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        fname = "paper_positions.json" if mode == TradingMode.PAPER else "real_positions.json"
        self._path: Path = log_dir / fname
        self._positions: list[Position] = self._load()

    def _load(self) -> list[Position]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text())
            return [Position.from_dict(d) for d in data]
        except Exception as e:
            logger.warning("Falha ao carregar posições: %s", e)
            return []

    def _save(self) -> None:
        try:
            self._path.write_text(
                json.dumps([p.to_dict() for p in self._positions], indent=2)
            )
        except Exception as e:
            logger.warning("Falha ao guardar posições: %s", e)

    def add(self, position: Position) -> None:
        self._positions.append(position)
        self._save()

    def open_positions(self) -> list[Position]:
        return [p for p in self._positions if p.status == PositionStatus.OPEN]

    def all_positions(self) -> list[Position]:
        return list(self._positions)

    def today_position(self) -> Position | None:
        today = date.today().isoformat()
        for p in reversed(self._positions):
            if p.date_opened == today:
                return p
        return None

    def refresh(self, clob_client: "ClobClient | None") -> None:
        from datetime import datetime as _dt
        now_str = _dt.now().isoformat()

        for pos in self.open_positions():
            mid = self._get_mid(pos, clob_client)
            if mid is not None:
                pos.current_mid = round(mid, 4)
                pos.pnl_usd = round((mid - pos.entry_ask) * pos.shares, 2)
                pos.pnl_pct = round((mid / pos.entry_ask - 1) * 100, 2) if pos.entry_ask else None

            if self.mode == TradingMode.REAL and pos.market_slug:
                resolved, won = self._check_resolution(pos)
                if resolved:
                    if won:
                        pos.status  = PositionStatus.WON
                        pos.pnl_usd = round((1.0 - pos.entry_ask) * pos.shares, 2)
                        pos.pnl_pct = round((1.0 / pos.entry_ask - 1) * 100, 2)
                    else:
                        pos.status  = PositionStatus.LOST
                        pos.pnl_usd = round(-pos.size_usdc, 2)
                        pos.pnl_pct = -100.0

            pos.last_updated = now_str

        self._save()

    def _get_mid(self, pos: Position,
                 clob_client: "ClobClient | None") -> float | None:
        if not clob_client or not pos.token_id:
            return None
        try:
            book = clob_client.get_orderbook(pos.token_id)
            if book:
                return book.best_bid or book.mid
        except Exception:
            pass
        return None

    def _check_resolution(self, pos: Position) -> tuple[bool, bool]:
        import requests as _req
        if not pos.market_slug:
            return False, False
        try:
            r = _req.get(
                f"{GAMMA_API}/events",
                params={"slug": pos.market_slug},
                timeout=10,
            )
            r.raise_for_status()
            events = r.json()
            if not events:
                return False, False
            event = events[0] if isinstance(events, list) else events

            for m in event.get("markets", []):
                token_ids_raw = m.get("clobTokenIds", "[]")
                if isinstance(token_ids_raw, str):
                    try:
                        token_ids_raw = json.loads(token_ids_raw)
                    except Exception:
                        token_ids_raw = []

                if pos.token_id not in (token_ids_raw or []):
                    continue

                if not m.get("resolved", False):
                    return False, False

                winner = str(m.get("winner") or "").lower()
                won = winner in ("yes", "true", "1")
                return True, won

        except Exception as e:
            logger.warning("_check_resolution falhou: %s", e)

        return False, False

    def pnl_summary(self) -> dict:
        invested = sum(p.size_usdc for p in self._positions)
        pnl_sum  = sum(p.pnl_usd for p in self._positions if p.pnl_usd is not None)
        return {
            "total_invested": round(invested, 2),
            "total_pnl_usd":  round(pnl_sum, 2),
            "total_pnl_pct":  round(pnl_sum / invested * 100, 2) if invested else 0.0,
            "n_open":    sum(1 for p in self._positions if p.status == PositionStatus.OPEN),
            "n_won":     sum(1 for p in self._positions if p.status == PositionStatus.WON),
            "n_lost":    sum(1 for p in self._positions if p.status == PositionStatus.LOST),
            "n_unknown": sum(1 for p in self._positions if p.status == PositionStatus.UNKNOWN),
        }
