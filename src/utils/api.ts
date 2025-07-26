// API utility functions for backend communication

const API_BASE_URL = 'http://localhost:8000'

export interface CallSessionData {
  cust_id: string
  start_time?: string
  end_time?: string
  duration_secs?: number
  positive?: number
  neutral?: number
  negative?: number
  key_words?: string
  summarized_content?: string
  customer_suggestions?: string
  admin_suggestions?: string
}

export interface CallSessionResponse {
  id: number
  cust_id: string
  start_time: string
  end_time: string
  duration_secs: number
  positive: number
  neutral: number
  negative: number
  key_words: string
  summarized_content: string
  customer_suggestions: string
  admin_suggestions: string
}

/**
 * Create a new call session
 */
export async function createCallSession(custId: string = 'anonymous'): Promise<CallSessionResponse> {
  try {
    const callSessionData: CallSessionData = {
      cust_id: custId,
      start_time: new Date().toISOString()
    }

    const response = await fetch(`${API_BASE_URL}/call_session/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(callSessionData),
    })

    if (!response.ok) {
      throw new Error(`Failed to create call session: ${response.status} ${response.statusText}`)
    }

    const result = await response.json()
    console.log('✅ Call session created:', result)
    return result
  } catch (error) {
    console.error('❌ Error creating call session:', error)
    throw error
  }
}

/**
 * Update a call session (e.g., when call ends)
 */
export async function updateCallSession(
  sessionId: number,
  updateData: Partial<CallSessionData>
): Promise<CallSessionResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/call_session/${sessionId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(updateData),
    })

    if (!response.ok) {
      throw new Error(`Failed to update call session: ${response.status} ${response.statusText}`)
    }

    const result = await response.json()
    console.log('✅ Call session updated:', result)
    return result
  } catch (error) {
    console.error('❌ Error updating call session:', error)
    throw error
  }
}

/**
 * Get call session by ID
 */
export async function getCallSession(sessionId: number): Promise<CallSessionResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/call_session/${sessionId}`)

    if (!response.ok) {
      throw new Error(`Failed to get call session: ${response.status} ${response.statusText}`)
    }

    const result = await response.json()
    return result
  } catch (error) {
    console.error('❌ Error getting call session:', error)
    throw error
  }
}