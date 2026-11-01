"""RGBIC segment colours read back with ``aa a5`` and shown on the segment entities.

The frames are real answers from an H706A (Permanent Outdoor Lights Pro, 13
segments) over the AWS IoT passthrough: every segment warm white at 100 %.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.govee.api.ble_packet import build_segment_read_packet
from custom_components.govee.const import CONF_SEGMENT_MODE_BY_DEVICE, SEGMENT_MODE_DISABLED
from custom_components.govee.coordinator import SEGMENT_READ_DELAY_SECONDS, SEGMENT_READ_MIN_INTERVAL
from custom_components.govee.models import GoveeCapability, GoveeDevice, GoveeDeviceState, RGBColor, SegmentState
from custom_components.govee.models.device import CAPABILITY_SEGMENT_COLOR
from custom_components.govee.platforms.segment import GoveeSegmentEntity
from tests.test_cov_coordinator_control import _POWER, _add, _coordinator

DEV = "1B:4B:C8:F5:AA:E6:1A:55"
WARM = (255, 174, 84)
PAGE_1 = bytes.fromhex("aaa50164ffae5464ffae5464ffae5464ffae540e")
PAGE_3 = bytes.fromhex("aaa50364ffae5464ffae5464ffae5464ffae540c")
PAGE_4 = bytes.fromhex("aaa504000000000000000000000000000000000b")


def _segments_cap(count: int) -> GoveeCapability:
    return GoveeCapability(
        type=CAPABILITY_SEGMENT_COLOR,
        instance="segmentedColorRgb",
        parameters={"fields": [{"fieldName": "segment", "elementRange": {"min": 0, "max": count - 1}}]},
    )


def _strip(count: int = 13, *, sku: str = "H6072") -> GoveeDevice:
    return GoveeDevice(
        device_id=DEV,
        sku=sku,
        name="Roof",
        device_type="devices.types.light",
        capabilities=(_POWER, _segments_cap(count)),
    )


def _frame(page: int, slots: list[tuple[int, int, int, int]]) -> bytes:
    body = bytes([0xAA, 0xA5, page]) + b"".join(bytes(slot) for slot in slots)
    body = body.ljust(19, b"\x00")
    checksum = 0
    for byte in body:
        checksum ^= byte
    return body + bytes([checksum])


class TestPacket:
    def test_matches_the_query_the_h706a_answered(self):
        assert build_segment_read_packet(1) == bytes.fromhex("aaa501" + "00" * 16 + "0e")
        assert build_segment_read_packet(2) == bytes.fromhex("aaa502" + "00" * 16 + "0d")


class TestDecoder:
    def test_real_pages(self):
        state = GoveeDeviceState.create_empty(DEV)
        assert state.update_segments_from_frames([PAGE_1, PAGE_3, PAGE_4], 13) is True
        assert [seg.index for seg in state.segments] == [0, 1, 2, 3, 8, 9, 10, 11, 12]
        assert state.segments[0] == SegmentState(index=0, color=RGBColor(*WARM), brightness=100)
        assert state.segments[-1] == SegmentState(index=12, color=RGBColor(0, 0, 0), brightness=0)

    def test_padding_past_segment_count_is_ignored(self):
        state = GoveeDeviceState.create_empty(DEV)
        state.update_segments_from_frames([PAGE_4], 13)
        assert [seg.index for seg in state.segments] == [12]

    def test_unchanged_returns_false(self):
        state = GoveeDeviceState.create_empty(DEV)
        state.update_segments_from_frames([PAGE_1], 13)
        assert state.update_segments_from_frames([PAGE_1], 13) is False

    def test_one_segment_changing_is_a_change(self):
        state = GoveeDeviceState.create_empty(DEV)
        state.update_segments_from_frames([PAGE_1], 13)
        frame = _frame(1, [(100, 255, 174, 84), (50, 0, 0, 255), (100, 255, 174, 84), (100, 255, 174, 84)])
        assert state.update_segments_from_frames([frame], 13) is True
        assert state.segments[1] == SegmentState(index=1, color=RGBColor(0, 0, 255), brightness=50)

    def test_invalid_frames_ignored(self):
        bad_sum = bytearray(PAGE_1)
        bad_sum[19] ^= 0xFF
        other = bytearray(PAGE_1)
        other[1] = 0x05
        page_zero = _frame(0, [(100, 1, 2, 3)] * 4)
        state = GoveeDeviceState.create_empty(DEV)
        frames = [bytes(bad_sum), PAGE_1[:19], PAGE_1 + b"\x00", bytes(other), page_zero, b""]
        assert state.update_segments_from_frames(frames, 13) is False
        assert state.segments == []


class TestMqttPush:
    def test_ptreal_answer_applies_and_notifies_without_scheduling(self):
        coord = _coordinator()
        state = _add(coord, _strip())
        with patch.object(coord, "_schedule_segment_read") as schedule:
            coord._on_mqtt_state_update(DEV, {"result": 1, "_op_frames": [PAGE_1.hex()]})
        assert len(state.segments) == 4
        coord.async_set_updated_data.assert_called_once()
        schedule.assert_not_called()

    def test_status_report_schedules_a_read(self):
        coord = _coordinator()
        _add(coord, _strip())
        with patch.object(coord, "_schedule_segment_read") as schedule:
            coord._on_mqtt_state_update(DEV, {"onOff": 1, "brightness": 35})
        schedule.assert_called_once_with(DEV)

    def test_segments_disabled_neither_reads_nor_decodes(self):
        coord = _coordinator(options={CONF_SEGMENT_MODE_BY_DEVICE: {DEV: SEGMENT_MODE_DISABLED}})
        state = _add(coord, _strip())
        with patch.object(coord, "_schedule_segment_read") as schedule:
            coord._on_mqtt_state_update(DEV, {"onOff": 1, "_op_frames": [PAGE_1.hex()]})
        assert state.segments == []
        schedule.assert_not_called()

    def test_device_without_segments_is_untouched(self):
        coord = _coordinator()
        device = GoveeDevice(
            device_id=DEV, sku="H6072", name="Lamp", device_type="devices.types.light", capabilities=(_POWER,)
        )
        state = _add(coord, device)
        with patch.object(coord, "_schedule_segment_read") as schedule:
            coord._on_mqtt_state_update(DEV, {"onOff": 1, "_op_frames": [PAGE_1.hex()]})
        assert state.segments == []
        schedule.assert_not_called()

    def test_group_is_not_read(self):
        coord = _coordinator()
        group = GoveeDevice(
            device_id=DEV,
            sku="SameModeGroup",
            name="Group",
            device_type="devices.types.light",
            capabilities=(_POWER, _segments_cap(4)),
            is_group=True,
        )
        assert coord._reads_segments(group) is False


def _ble(coord: Any, *, available: bool = True, sent: bool = True) -> AsyncMock:
    send = AsyncMock(return_value=sent)
    coord._ble_manager = MagicMock(available=available, async_send_ble_packet=send)
    return send


class TestRead:
    @pytest.mark.asyncio
    async def test_one_query_per_four_segments(self):
        coord = _coordinator()
        _add(coord, _strip(13))
        send = _ble(coord)
        with patch("custom_components.govee.coordinator.asyncio.sleep", AsyncMock()):
            assert await coord.async_read_segments(DEV) is True
        packets = [call.args[2] for call in send.await_args_list]
        assert len(packets) == 4
        assert packets[0] == "qqUBAAAAAAAAAAAAAAAAAAAAAA4="  # aa a5 01, base64

    @pytest.mark.asyncio
    async def test_stops_when_a_publish_fails(self):
        coord = _coordinator()
        _add(coord, _strip(13))
        send = _ble(coord, sent=False)
        with patch("custom_components.govee.coordinator.asyncio.sleep", AsyncMock()):
            assert await coord.async_read_segments(DEV) is False
        assert send.await_count == 1

    @pytest.mark.asyncio
    async def test_nothing_without_passthrough_or_device(self):
        coord = _coordinator()
        _add(coord, _strip(13))
        send = _ble(coord, available=False)
        assert await coord.async_read_segments(DEV) is False
        assert await coord.async_read_segments("unknown") is False
        send.assert_not_awaited()


class TestSchedule:
    def test_debounced_and_rate_limited(self):
        coord = _coordinator()
        _add(coord, _strip())
        cancel = MagicMock()
        with patch("custom_components.govee.coordinator.async_call_later", return_value=cancel) as later:
            coord._schedule_segment_read(DEV)
            coord._schedule_segment_read(DEV)
            assert later.call_count == 1
            assert later.call_args.args[1] == SEGMENT_READ_DELAY_SECONDS

            # The read fires: it clears the pending marker and reads.
            with patch.object(coord, "async_read_segments", AsyncMock(return_value=True)) as read:
                asyncio.run(later.call_args.args[2](None))
            read.assert_awaited_once_with(DEV)
            assert DEV not in coord._segment_read_unsubs

            # A read that just went out holds the next one off.
            coord._segment_read_last[DEV] = 1000.0
            with patch("custom_components.govee.coordinator.time.monotonic", return_value=1000.0 + 1):
                coord._schedule_segment_read(DEV)
            assert later.call_count == 1
            with patch(
                "custom_components.govee.coordinator.time.monotonic",
                return_value=1000.0 + SEGMENT_READ_MIN_INTERVAL + 1,
            ):
                coord._schedule_segment_read(DEV)
            assert later.call_count == 2

    @pytest.mark.asyncio
    async def test_shutdown_cancels_a_pending_read(self):
        coord = _coordinator()
        cancel = MagicMock()
        coord._segment_read_unsubs[DEV] = cancel
        coord._api_client.close = AsyncMock()
        await coord.async_shutdown()
        cancel.assert_called_once()
        assert coord._segment_read_unsubs == {}


def _entity(index: int, segments: list[SegmentState]) -> GoveeSegmentEntity:
    coordinator = MagicMock()
    coordinator.last_update_success = True
    state = GoveeDeviceState.create_empty(DEV)
    state.power_state = True
    state.segments = segments
    coordinator.get_state = MagicMock(return_value=state)
    entity = GoveeSegmentEntity(coordinator, _strip(), index)
    entity.hass = MagicMock()
    entity.async_write_ha_state = MagicMock()
    return entity


class TestSegmentEntity:
    def test_takes_the_reported_colour(self):
        entity = _entity(1, [SegmentState(index=1, color=RGBColor(0, 0, 255), brightness=50)])
        entity._handle_coordinator_update()
        assert entity.is_on is True
        assert entity.rgb_color == (0, 0, 255)
        assert entity.brightness == 128
        entity.async_write_ha_state.assert_called()

    def test_unchanged_report_keeps_a_newer_local_write(self):
        entity = _entity(1, [SegmentState(index=1, color=RGBColor(0, 0, 255), brightness=100)])
        entity._handle_coordinator_update()
        entity._rgb_color = (255, 0, 0)  # what async_turn_on just set
        entity._handle_coordinator_update()
        assert entity.rgb_color == (255, 0, 0)

    def test_black_or_dark_segment_is_off_and_keeps_its_colour(self):
        entity = _entity(12, [SegmentState(index=12, color=RGBColor(0, 0, 0), brightness=0)])
        entity._rgb_color = (10, 20, 30)
        entity._handle_coordinator_update()
        assert entity.is_on is False
        assert entity.rgb_color == (10, 20, 30)

    def test_no_report_leaves_optimistic_state(self):
        entity = _entity(5, [SegmentState(index=1, color=RGBColor(0, 0, 255), brightness=100)])
        entity._handle_coordinator_update()
        assert entity.is_on is True
        assert entity.rgb_color == (255, 255, 255)

    def test_no_state_leaves_optimistic_state(self):
        entity = _entity(1, [])
        entity.coordinator.get_state = MagicMock(return_value=None)
        entity._handle_coordinator_update()
        assert entity.rgb_color == (255, 255, 255)


def test_ptreal_segment_fixtures_are_not_read():
    coord = _coordinator()
    assert coord._reads_segments(_strip(17, sku="H1232")) is False
    assert coord._reads_segments(_strip(13, sku="H706A")) is True
