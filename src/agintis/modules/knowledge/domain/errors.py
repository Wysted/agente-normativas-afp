"""Domain errors of the knowledge module."""

from agintis.shared_kernel.errors import DomainError


class InvalidMetadataError(DomainError):
    """Chunk metadata is reserved, malformed or does not fit the profile's declared schema."""


class DuplicateDocumentError(DomainError):
    """A READY document already exists for the same (tenant, profile, source_uri).

    SQL adapters translate the partial unique index violation
    (UNIQUE(tenant_id, profile, source_uri) WHERE status = 'ready') into this error.
    """


class TenantMismatchError(DomainError):
    """An upsert targeted an existing document id owned by a different tenant."""


class DocumentIdentityError(DomainError):
    """An upsert tried to change the identity (profile, source_uri) of an existing document id.

    A document id is bound to its (tenant, profile, source_uri) forever; a new version of a
    source is a NEW document id.
    """


class DocumentNotFoundError(DomainError):
    """Chunks were added to a document that does not exist FOR THE CHUNKS' TENANT AND PROFILE.

    Deliberately one error for "missing" and "owned by another tenant": telling them apart
    would reveal that a foreign tenant has a document with that id.
    """
